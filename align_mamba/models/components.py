"""Core components for Align-Mamba models.

Contains:
- RMSNorm: Required for Mamba stability at scale
- ScaledEmbedding: Token embedding with learnable scale and dropout
- RotaryPositionalEmbedding (RoPE): Applied ONLY to attention layers, NOT Mamba
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from align_mamba.kernels.rmsnorm import fused_rmsnorm


class RMSNorm(nn.Module):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.

    Features:
    - Fused Triton kernel on CUDA for efficiency
    - Adaptive epsilon based on dtype (prevents BF16/FP16 underflow)
    """

    def __init__(
        self,
        d_model: int,
        eps: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model

        if eps is None:
            if dtype in (torch.bfloat16, torch.float16):
                self.eps = 1e-4
            else:
                self.eps = 1e-6
        else:
            self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "RMSNorm requires CUDA tensors"
        return fused_rmsnorm(x, self.weight, self.eps)


class ScaledEmbedding(nn.Module):
    """Token embedding with learnable scale and dropout.

    The scale is initialized to sqrt(d_model) following the Transformer convention,
    but is learnable to allow the model to adjust embedding magnitudes during training.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 0,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx, device=device)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(d_model)))
        self.embed_dropout = nn.Dropout(dropout)
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        return self.embed_dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding for attention layers.

    Applies rotation to Q and K tensors to encode relative positions.
    Supports dynamic cache resizing for length extrapolation.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _extend_cache(self, seq_len: int):
        if seq_len > self.max_seq_len:
            new_max = max(seq_len, self.max_seq_len * 2)
            self._build_cache(new_max, device=self.inv_freq.device)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_to_qk_bthd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_offset: int = 0,
        k_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings in (B, T, H, D) layout without transposes.

        This method avoids 4 transpose operations per attention forward by
        directly operating in BTHD layout used by FlashAttention.

        Args:
            q: Query tensor (B, T, H, D)
            k: Key tensor (B, T_k, H, D)
            q_offset: Position offset for queries (for autoregressive decoding)
            k_offset: Position offset for keys (typically 0)

        Returns:
            Tuple of rotated (q, k) in same (B, T, H, D) layout
        """
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        required_len = max(q_offset + seq_len_q, k_offset + seq_len_k)
        if required_len > self.max_seq_len:
            self._extend_cache(required_len)

        # Reshape for BTHD: (T, D) -> (1, T, 1, D)
        cos_q = self.cos_cache[q_offset : q_offset + seq_len_q].unsqueeze(0).unsqueeze(2).to(q.dtype)
        sin_q = self.sin_cache[q_offset : q_offset + seq_len_q].unsqueeze(0).unsqueeze(2).to(q.dtype)
        cos_k = self.cos_cache[k_offset : k_offset + seq_len_k].unsqueeze(0).unsqueeze(2).to(k.dtype)
        sin_k = self.sin_cache[k_offset : k_offset + seq_len_k].unsqueeze(0).unsqueeze(2).to(k.dtype)

        q_rot = (q * cos_q) + (self._rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (self._rotate_half(k) * sin_k)

        return q_rot, k_rot
