# src/models/text_encoder.py
#
# Frozen BioClinicalBERT encoder for extracting clinical text embeddings.
# Used during EchoJEPAv2 pretraining to condition the predictor on study-level
# clinical text (echo report, measurements, diagnoses).
#
# The BERT weights are always frozen; only the downstream text projection
# (housed in the predictor) is trained.

from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)

_BIOCLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


class BioClinicalBERTEncoder(nn.Module):
    """
    Thin wrapper around BioClinicalBERT that returns per-token hidden states.

    Weights are frozen at init time. The module is always in eval mode for
    inference but still participates in the computation graph (frozen gradients
    flow through but BERT parameters receive no gradient updates).

    Args:
        model_name: HuggingFace model identifier (default: Bio_ClinicalBERT).
        output_dim: hidden size of the returned embeddings (768 for BERT-base).

    Returns (forward):
        token_embeddings: (B, L, hidden_size) float32 — all token hidden states
                          from the last BERT layer, including [CLS] at index 0.
    """

    def __init__(self, model_name: str = _BIOCLINICALBERT_MODEL):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers is required for text conditioning. "
                "Install with: pip install transformers"
            )

        logger.info(f"Loading BioClinicalBERT from '{model_name}'")
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.bert.config.hidden_size  # 768 for BERT-base

        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        logger.info(
            f"BioClinicalBERT loaded ({self.hidden_size}-dim hidden). "
            "All weights frozen."
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L) LongTensor
        attention_mask: torch.Tensor,  # (B, L) LongTensor — 1=real, 0=padding
    ) -> torch.Tensor:
        """
        Returns token-level embeddings from the last BERT layer.

        Output shape: (B, L, hidden_size)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, hidden_size)
