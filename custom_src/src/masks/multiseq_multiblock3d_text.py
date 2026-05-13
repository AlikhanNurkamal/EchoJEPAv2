# src/masks/multiseq_multiblock3d_text.py
#
# Text-aware MaskCollator for EchoJEPAv2 text-conditioned pretraining.
#
# Extends MaskCollator to handle the 4-tuple (clips, label, clip_indices, text_str)
# returned by WebDatasetVideoDatasetText.  The collator owns a HuggingFace
# tokeniser (loaded once in the main process) and converts raw text strings into
# padded token tensors before returning the batch.
#
# Output per fpc_collation:
#   (collated_batch, collated_masks_enc, collated_masks_pred)
#   collated_batch[0] — clips tensor
#   collated_batch[1] — labels tensor
#   collated_batch[2] — clip_indices tensor
#   collated_batch[3] — text_input_ids    (B, max_text_len) LongTensor
#   collated_batch[4] — text_attention_mask (B, max_text_len) LongTensor

import math
from logging import getLogger
from multiprocessing import Value

import torch

logger = getLogger()

_DEFAULT_TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
_DEFAULT_MAX_TEXT_LEN = 256


class MaskCollatorText:
    """
    Drop-in replacement for MaskCollator that additionally tokenises
    clinical text strings in the main process.

    Args:
        cfgs_mask:          list of mask-generator configs (same as MaskCollator).
        dataset_fpcs:       list of frames-per-clip values.
        crop_size:          spatial crop size in pixels.
        patch_size:         ViT patch size in pixels.
        tubelet_size:       temporal patch (tubelet) size in frames.
        text_model_name:    HuggingFace model name for the tokeniser.
        max_text_len:       maximum number of tokens (padding/truncation length).
    """

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        text_model_name: str = _DEFAULT_TEXT_MODEL,
        max_text_len: int = _DEFAULT_MAX_TEXT_LEN,
    ):
        super().__init__()

        self.max_text_len = max_text_len

        # Build mask generators (identical to the original MaskCollator)
        self.mask_generators: dict[int, list] = {}
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                mask_generator = _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                )
                self.mask_generators[fpc].append(mask_generator)

        # Load tokeniser in the main process — never pickled into workers
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            logger.info(
                f"MaskCollatorText: loaded tokeniser from '{text_model_name}', "
                f"max_text_len={max_text_len}"
            )
        except ImportError:
            raise ImportError(
                "transformers is required for text conditioning. "
                "Install with: pip install transformers"
            )

    def step(self):
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    def __call__(self, batch):
        """
        Args:
            batch: list of (clips, label, clip_indices, text_str) tuples.

        Returns:
            fpc_collations: list of (collated_batch, masks_enc, masks_pred)
                collated_batch is a tuple:
                    [0] clips  [1] labels  [2] clip_indices
                    [3] text_input_ids (B, L)  [4] text_attention_mask (B, L)
        """
        # Split by frames-per-clip
        filtered_batches: dict[int, list] = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            fpc = len(sample[2][-1])  # clip_indices[-1] has length fpc
            filtered_batches[fpc].append(sample)

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue

            # Separate video data from text strings
            video_samples = [(s[0], s[1], s[2]) for s in fpc_batch]
            text_strings = [s[3] for s in fpc_batch]

            # Collate video/label/indices normally
            collated_video = torch.utils.data.default_collate(video_samples)

            # Tokenise text in the main process
            encoding = self.tokenizer(
                text_strings,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            text_input_ids = encoding["input_ids"]          # (B, L) LongTensor
            text_attention_mask = encoding["attention_mask"]  # (B, L) LongTensor

            # Combine into a single batch tuple
            collated_batch = tuple(collated_video) + (text_input_ids, text_attention_mask)

            # Generate masks
            collated_masks_pred, collated_masks_enc = [], []
            for mask_generator in self.mask_generators[fpc]:
                masks_enc, masks_pred = mask_generator(batch_size)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)

            fpc_collations.append((collated_batch, collated_masks_enc, collated_masks_pred))

        return fpc_collations


# ── Mask generator (verbatim copy from multiseq_multiblock3d.py) ──────────────


class _MaskGenerator:

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
    ):
        super().__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.crop_size = crop_size
        self.height, self.width = [crop_size[i] // spatial_patch_size[i] for i in (0, 1)]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )
        self.max_keep = max_keep
        self._itr_counter = Value("i", -1)
        self.inv_block = inv_block

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        return mask

    def __call__(self, batch_size):
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):

            empty_context = True
            while empty_context:

                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]

        if self.full_complement:
            collated_masks_pred = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc
        else:
            return collated_masks_enc, collated_masks_pred
