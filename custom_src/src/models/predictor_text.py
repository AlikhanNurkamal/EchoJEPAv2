# src/models/predictor_text.py
#
# Text-conditioned variant of the V-JEPA predictor.
#
# Adds a GatedTextCrossAttention block after the self-attention predictor
# blocks.  This allows the predictor to consult frozen BioClinicalBERT token
# embeddings when predicting masked video patches, creating implicit video–text
# alignment without changing the V-JEPA patch-level L1 loss.
#
# Architecture:
#   predictor_embed  : Linear(embed_dim → pred_embed_dim)
#   predictor_blocks : N × TransformerBlock (self-attention, unchanged)
#   text_cross_attn  : GatedTextCrossAttention   ← NEW
#   predictor_norm   : LayerNorm(pred_embed_dim)
#   predictor_proj   : Linear(pred_embed_dim → embed_dim)
#
# GatedTextCrossAttention:
#   - Projects BioClinicalBERT token embeddings (768-d) to pred_embed_dim
#   - Cross-attention: predictor tokens attend to (projected) text tokens
#   - Learned sigmoid gate controls per-token text contribution
#   - Residual connection + LayerNorm
#
# The video encoder (ViT) and the V-JEPA loss are completely unchanged.
# During downstream evaluation the video encoder is used directly; text is
# not required.

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import repeat_interleave_batch, trunc_normal_


# ── Gated cross-attention ─────────────────────────────────────────────────────


class GatedTextCrossAttention(nn.Module):
    """
    Predictor tokens (queries) attend to BioClinicalBERT token embeddings
    (keys & values) via multi-head cross-attention.  A learned sigmoid gate
    controls how much attended text is incorporated per predictor token.

    Forward:
        attn_out = cross_attn(query=x, key=text_kv, value=text_kv)
        gate     = sigmoid(W_g · attn_out)
        output   = LayerNorm(x + gate ⊙ attn_out)

    The gate being a function of attn_out (rather than x alone) means it
    can shut off text completely when the attended representation carries no
    useful signal.

    Args:
        video_dim:   predictor embedding dimension (pred_embed_dim).
        text_dim:    BioClinicalBERT hidden size (768 for BERT-base).
        num_heads:   number of cross-attention heads.
    """

    def __init__(
        self,
        video_dim: int,
        text_dim: int = 768,
        num_heads: int = 8,
    ):
        super().__init__()
        assert video_dim % num_heads == 0, (
            f"video_dim ({video_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Project text from BERT space to predictor space
        self.text_proj = nn.Linear(text_dim, video_dim, bias=False)

        # Cross-attention: video tokens query text tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=video_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Gate: sigmoid over the attended text representation
        self.gate_proj = nn.Linear(video_dim, video_dim, bias=True)
        self.gate_act = nn.Sigmoid()

        self.norm = nn.LayerNorm(video_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        x: torch.Tensor,                              # (B, N, video_dim)
        text_emb: torch.Tensor,                       # (B, L, text_dim)
        text_key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) bool, True=ignore
    ) -> torch.Tensor:
        """
        Returns updated predictor tokens of shape (B, N, video_dim).
        """
        # Project text embeddings into video/predictor space
        text_kv = self.text_proj(text_emb)  # (B, L, video_dim)

        # Cross-attention: video tokens attend to text tokens
        attn_out, _ = self.cross_attn(
            query=x,
            key=text_kv,
            value=text_kv,
            key_padding_mask=text_key_padding_mask,  # True = ignore that token
        )  # (B, N, video_dim)

        # Gated residual
        gate = self.gate_act(self.gate_proj(attn_out))  # (B, N, video_dim)
        return self.norm(x + gate * attn_out)


# ── Text-conditioned predictor ────────────────────────────────────────────────


class VisionTransformerPredictorText(nn.Module):
    """
    V-JEPA predictor augmented with gated text cross-attention.

    Identical to VisionTransformerPredictor (predictor.py) except:
      - Constructor accepts `text_dim` and `text_cross_attn_heads`.
      - A GatedTextCrossAttention layer is inserted after predictor_blocks.
      - `forward` accepts an optional `text_emb` tensor.

    When `text_emb` is None the module behaves identically to the original
    predictor (the cross-attention block is skipped).
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=False,
        # Text conditioning
        text_dim: int = 768,
        text_cross_attn_heads: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.chop_last_n_tokens = chop_last_n_tokens

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for _ in range(num_mask_tokens)]
            )

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
            )
        else:
            self.num_patches = num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.uniform_power = uniform_power

        self.predictor_pos_embed = None
        if not use_rope:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
            )

        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # ── Text cross-attention (new) ────────────────────────────────────────
        self.text_cross_attn = GatedTextCrossAttention(
            video_dim=predictor_embed_dim,
            text_dim=text_dim,
            num_heads=text_cross_attn_heads,
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # Initialise
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(
        self,
        x,
        masks_x,
        masks_y,
        mask_index: int = 1,
        has_cls: bool = False,
        text_emb: Optional[torch.Tensor] = None,           # (B, L, text_dim)
        text_key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) bool
    ):
        """
        Args:
            x:                    context tokens from the encoder.
            masks_x:              encoder mask indices.
            masks_y:              predictor (target) mask indices.
            text_emb:             BioClinicalBERT token embeddings, (B, L, text_dim).
                                  If None, text cross-attention is skipped.
            text_key_padding_mask: True at padding positions (B, L).
        """
        assert (masks_x is not None) and (masks_y is not None)
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)
        if has_cls:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
        _, N_ctxt, D = x.shape

        if not self.use_rope:
            x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embed, masks_x)

        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)
        if not self.use_rope:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_y)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens += pos_embs

        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x, masks_y], dim=1)

        argsort = torch.argsort(masks, dim=1)
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

        if self.chop_last_n_tokens > 0:
            x = x[:, : -self.chop_last_n_tokens]
            masks = masks[:, : -self.chop_last_n_tokens]

        if has_cls:
            x = torch.cat([x_cls, x], dim=1)

        # Self-attention blocks (unchanged)
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, masks, None, use_reentrant=False)
            else:
                x = blk(x, mask=masks, attn_mask=None)

        # ── Gated text cross-attention ────────────────────────────────────────
        if text_emb is not None:
            # x may have a repeat(len(masks_x)) batch dimension.
            # Expand text_emb to match.
            n_repeat = x.shape[0] // B
            if n_repeat > 1:
                text_emb_expanded = text_emb.repeat_interleave(n_repeat, dim=0)
                mask_expanded = (
                    text_key_padding_mask.repeat_interleave(n_repeat, dim=0)
                    if text_key_padding_mask is not None else None
                )
            else:
                text_emb_expanded = text_emb
                mask_expanded = text_key_padding_mask

            x = self.text_cross_attn(x, text_emb_expanded, mask_expanded)

        x = self.predictor_norm(x)

        if has_cls:
            x = x[:, 1:, :]

        if not self.return_all_tokens:
            reverse_argsort = torch.argsort(argsort, dim=1)
            x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
            x = x[:, N_ctxt:]

        x = self.predictor_proj(x)
        return x


def vit_predictor_text(**kwargs):
    model = VisionTransformerPredictorText(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
