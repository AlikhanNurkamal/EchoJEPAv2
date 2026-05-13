# app/vjepa_text/utils_text.py
#
# Initialisation helpers for text-conditioned V-JEPA pretraining.
# Mirrors app/vjepa/utils.py but instantiates:
#   - VisionTransformerPredictorText  (predictor with gated text cross-attention)
#   - PredictorMultiSeqWrapperText    (wrapper that passes text_emb through)
# The video encoder (MultiSeqWrapper + ViT) is identical to the baseline.

import logging
import re
import sys

import torch

import src.models.vision_transformer as video_vit
from src.models.predictor_text import vit_predictor_text
from src.models.text_encoder import BioClinicalBERTEncoder
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule
from src.utils.wrappers import MultiSeqWrapper
from src.utils.wrappers_text import PredictorMultiSeqWrapperText

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _strip_ddp_prefix(state_dict: dict) -> dict:
    return {re.sub(r"^module\.(?:module\.)?", "", k): v for k, v in state_dict.items()}


def init_video_model_text(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_large",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    # Text conditioning
    text_dim: int = 768,
    text_cross_attn_heads: int = 8,
):
    """
    Build encoder + text-conditioned predictor.

    The encoder is identical to the baseline (plain ViT wrapped in
    MultiSeqWrapper).  The predictor is VisionTransformerPredictorText
    wrapped in PredictorMultiSeqWrapperText.
    """
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    encoder = MultiSeqWrapper(encoder)

    predictor = vit_predictor_text(
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        text_dim=text_dim,
        text_cross_attn_heads=text_cross_attn_heads,
    )
    predictor = PredictorMultiSeqWrapperText(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    logger.info(f"Encoder parameters:   {count_parameters(encoder):,}")
    logger.info(f"Predictor parameters: {count_parameters(predictor):,}")

    return encoder, predictor


def init_text_encoder(
    device,
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
) -> BioClinicalBERTEncoder:
    """Load frozen BioClinicalBERT onto device."""
    text_encoder = BioClinicalBERTEncoder(model_name=text_model_name)
    text_encoder.to(device)
    text_encoder.eval()
    logger.info(f"BioClinicalBERT loaded on {device}. Parameters are frozen.")
    return text_encoder


def load_checkpoint_text(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    """Load a checkpoint saved by the text-conditioned training loop."""
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]
    itr = checkpoint.get("itr", 0)

    pretrained_dict = _strip_ddp_prefix(checkpoint["encoder"])
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f"Loaded encoder from epoch {epoch}: {msg}")

    pretrained_dict = _strip_ddp_prefix(checkpoint["predictor"])
    msg = predictor.load_state_dict(pretrained_dict)
    logger.info(f"Loaded predictor from epoch {epoch}: {msg}")

    if target_encoder is not None:
        pretrained_dict = _strip_ddp_prefix(checkpoint["target_encoder"])
        msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f"Loaded target encoder from epoch {epoch}: {msg}")

    opt.load_state_dict(checkpoint["opt"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"Loaded optimiser from epoch {epoch}")
    del checkpoint

    return encoder, predictor, target_encoder, opt, scaler, epoch, itr


def init_opt_text(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    """
    Build AdamW optimizer over encoder + predictor parameters.

    BioClinicalBERT weights are frozen (requires_grad=False) and therefore
    naturally excluded from the parameter groups below — no special handling needed.
    The GatedTextCrossAttention and text projection layers inside the predictor
    ARE trainable and are included via predictor.named_parameters().
    """
    param_groups = [
        {
            "params": (
                p for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
