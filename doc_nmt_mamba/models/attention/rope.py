"""
Rotary Positional Embedding (RoPE) for attention layers.

IMPORTANT: Apply RoPE ONLY to attention layers, NOT to Mamba layers.
- Mamba encodes position implicitly through the recurrence
- Attention is permutation-invariant without positional encoding
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding for attention layers.

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
        """
        Args:
            dim: Head dimension (d_model // n_heads)
            max_seq_len: Initial maximum sequence length to cache
            base: Base for frequency computation
            device: Device for cached tensors
        """
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build initial cache
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos/sin caches for efficiency."""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Duplicate for real/imaginary pairing
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _extend_cache(self, seq_len: int):
        """Dynamically extend cache for longer sequences (length extrapolation)."""
        if seq_len > self.max_seq_len:
            # Double the cache size or extend to seq_len, whichever is larger
            new_max = max(seq_len, self.max_seq_len * 2)
            self._build_cache(new_max, device=self.inv_freq.device)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to Q and K.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)
            offset: Position offset for incremental decoding

        Returns:
            Tuple of rotated (q, k)
        """
        seq_len = q.size(2)

        # Dynamically extend cache if needed (length extrapolation support)
        required_len = offset + seq_len
        if required_len > self.max_seq_len:
            self._extend_cache(required_len)

        # Get cached cos/sin for current positions and match input dtype
        cos = self.cos_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_offset: int = 0,
        k_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings with separate offsets for Q and K.
        Useful for cross-attention where Q and K may have different lengths.

        Args:
            q: Query tensor (batch, n_heads, seq_len_q, head_dim)
            k: Key tensor (batch, n_heads, seq_len_k, head_dim)
            q_offset: Position offset for queries
            k_offset: Position offset for keys

        Returns:
            Tuple of rotated (q, k)
        """
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        # Dynamically extend cache if needed (length extrapolation support)
        required_len = max(q_offset + seq_len_q, k_offset + seq_len_k)
        if required_len > self.max_seq_len:
            self._extend_cache(required_len)

        # Get cached cos/sin and convert to input dtype
        cos_q = self.cos_cache[q_offset : q_offset + seq_len_q].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin_q = self.sin_cache[q_offset : q_offset + seq_len_q].unsqueeze(0).unsqueeze(0).to(q.dtype)

        cos_k = self.cos_cache[k_offset : k_offset + seq_len_k].unsqueeze(0).unsqueeze(0).to(k.dtype)
        sin_k = self.sin_cache[k_offset : k_offset + seq_len_k].unsqueeze(0).unsqueeze(0).to(k.dtype)

        q_rot = (q * cos_q) + (self._rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (self._rotate_half(k) * sin_k)

        return q_rot, k_rot

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"
