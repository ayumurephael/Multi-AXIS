from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flash_only_context(require_flash: bool):
    if not torch.cuda.is_available():
        return nullcontext()
    if not require_flash:
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    except (ImportError, AttributeError):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        )


def _flash_fixed_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Run dense non-causal cross-attention with the FlashAttention-2 package."""
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError) as exc:
        raise RuntimeError("CUDA cross-attention requires the flash-attn package") from exc
    output = flash_attn_func(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        dropout_p=0.0,
        causal=False,
    )
    return output.transpose(1, 2)


def _flash_varlen_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Run masked non-causal cross-attention with FlashAttention varlen packing."""
    try:
        from flash_attn import flash_attn_varlen_func
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Masked CUDA cross-attention requires the flash-attn package"
        ) from exc

    batch, _, q_len, _ = q.shape
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    q_unpadded = q_bshd.view(batch * q_len, q.shape[1], q.shape[-1])
    k_unpadded = k_bshd[key_padding_mask]
    v_unpadded = v_bshd[key_padding_mask]
    key_lengths = key_padding_mask.sum(dim=-1, dtype=torch.int32)
    cu_q = torch.arange(
        0, (batch + 1) * q_len, q_len, device=q.device, dtype=torch.int32
    )
    cu_k = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
    cu_k[1:] = torch.cumsum(key_lengths, dim=0)
    output = flash_attn_varlen_func(
        q_unpadded,
        k_unpadded,
        v_unpadded,
        cu_q,
        cu_k,
        q_len,
        int(key_lengths.max().item()),
        dropout_p=0.0,
        causal=False,
    )
    return output.view(batch, q_len, q.shape[1], q.shape[-1]).transpose(1, 2)


class FlashCrossAttention(nn.Module):
    """Non-causal MHA whose CUDA path is forced onto the FlashAttention-2 SDPA kernel."""

    def __init__(self, embed_dim: int, num_heads: int, require_flash: bool = True):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.require_flash = require_flash
        if require_flash and self.head_dim > 256:
            raise ValueError(
                "FlashAttention-2 requires head_dim <= 256; "
                f"received embed_dim={embed_dim}, num_heads={num_heads}, "
                f"head_dim={self.head_dim}"
            )
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query/key/value must have shape [B, N, D]")
        if key.shape != value.shape or query.shape[0] != key.shape[0]:
            raise ValueError("Cross-attention batch/key/value shapes do not match")
        batch, q_len, _ = query.shape
        kv_len = key.shape[1]

        def split(x: torch.Tensor, length: int) -> torch.Tensor:
            return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(query), q_len)
        k = split(self.k_proj(key), kv_len)
        v = split(self.v_proj(value), kv_len)
        attention_mask = None
        has_padding = False
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, kv_len):
                raise ValueError("key_padding_mask must have shape [B, K]")
            if not bool(key_padding_mask.any(dim=-1).all()):
                raise ValueError("Every sample must expose at least one valid key")
            key_padding_mask = key_padding_mask.bool()
            has_padding = not bool(key_padding_mask.all())
            if not has_padding:
                key_padding_mask = None

        if q.is_cuda and self.require_flash:
            output = (
                _flash_varlen_cross_attention(q, k, v, key_padding_mask)
                if has_padding
                else _flash_fixed_cross_attention(q, k, v)
            )
        else:
            if key_padding_mask is not None:
                attention_mask = key_padding_mask[:, None, None, :]
            with _flash_only_context(self.require_flash):
                output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        output = output.transpose(1, 2).contiguous().view(batch, q_len, self.embed_dim)
        return self.out_proj(output)
