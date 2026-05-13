# src/utils/wrappers_text.py
#
# Text-aware wrapper for VisionTransformerPredictorText.
#
# Mirrors PredictorMultiSeqWrapper (wrappers.py) but threads optional
# `text_emb` and `text_key_padding_mask` arguments through to the underlying
# predictor's forward, enabling gated cross-attention with clinical text.
#
# MultiSeqWrapper is identical to the original and is re-exported here for
# convenience so the text training script only needs to import from one place.

from typing import Optional

import torch
import torch.nn as nn

from src.utils.wrappers import MultiSeqWrapper  # re-export unchanged


class PredictorMultiSeqWrapperText(nn.Module):
    """
    Wraps VisionTransformerPredictorText to handle the list-of-lists output
    from MultiSeqWrapper and pass text embeddings to each predictor call.

    Args:
        backbone: a VisionTransformerPredictorText instance.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        x,
        masks_x,
        masks_y,
        has_cls: bool = False,
        text_emb: Optional[torch.Tensor] = None,           # (B, L, text_dim)
        text_key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) bool
    ):
        """
        Args:
            x:        list-of-lists of encoder output tensors
                      (outer = per-FPC, inner = per-mask).
            masks_x:  list-of-lists of encoder mask tensors.
            masks_y:  list-of-lists of predictor mask tensors.
            text_emb: BioClinicalBERT embeddings (B, L, text_dim).
                      Broadcast to every backbone call.  If None, text
                      cross-attention is skipped (graceful degradation).
            text_key_padding_mask: True at padding positions (B, L).

        Returns:
            outs: list-of-lists matching the shape of x, each element is a
                  predicted patch tensor.
        """
        outs = [[] for _ in x]
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mxij, myij in zip(xi, mxi, myi):
                outs[i].append(
                    self.backbone(
                        xij,
                        mxij,
                        myij,
                        mask_index=i,
                        has_cls=has_cls,
                        text_emb=text_emb,
                        text_key_padding_mask=text_key_padding_mask,
                    )
                )
        return outs
