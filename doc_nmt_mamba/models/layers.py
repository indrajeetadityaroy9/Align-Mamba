"""
Consolidated Building Blocks for Hybrid Mamba-Attention Architecture.

This file contains all layer components:
- RMSNorm: Stable normalization for 200M+ params
- RotaryPositionalEmbedding: RoPE for attention layers only
- segment_aware_flip: Document-boundary-respecting sequence flip
- Attention layers: Bidirectional, Causal, Cross-attention
- Mamba blocks: Mamba2BlockWrapper, BiMambaBlock

IMPORTANT: Apply RoPE ONLY to attention layers, NOT to Mamba layers.
Mamba encodes position implicitly through recurrence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# =============================================================================
# CUDA-dependent imports with fallbacks
# H100 Kernel Selection: Prefer Dao-AI-Lab CUDA kernels for 10-50x speedup
# =============================================================================

# Mamba-2 (CUDA only - CRITICAL for H100 performance)
_mamba2_available = False
_mamba2_optimized_kernels = False
Mamba2 = None

try:
    from mamba_ssm import Mamba2 as _Mamba2
    Mamba2 = _Mamba2
    _mamba2_available = True

    # Check for optimized Triton kernels (critical for H100 performance)
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        _mamba2_optimized_kernels = True
    except ImportError:
        _mamba2_optimized_kernels = False
        warnings.warn(
            "mamba-ssm Triton kernels not available. "
            "Performance on H100 will be suboptimal. "
            "Reinstall with: pip install mamba-ssm --no-build-isolation"
        )
except ImportError:
    warnings.warn(
        "CRITICAL: mamba-ssm not available. Mamba2BlockWrapper and BiMambaBlock will not work. "
        "On H100, this means 10-50x slower training! "
        "Install with: pip install mamba-ssm causal-conv1d --no-build-isolation (requires CUDA)"
    )

# FlashAttention-2 (optional, falls back to PyTorch SDPA)
# H100 has native SDPA support, but FlashAttention-2 is still faster
FLASH_ATTN_AVAILABLE = False
flash_attn_func = None
flash_attn_varlen_func = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    # PyTorch SDPA fallback is acceptable on H100 (native support)
    pass


def check_h100_kernel_status():
    """
    Check and report H100 kernel optimization status.

    Call this at training start to ensure optimal configuration.
    """
    status = {
        "mamba2_available": _mamba2_available,
        "mamba2_optimized": _mamba2_optimized_kernels,
        "flash_attn_available": FLASH_ATTN_AVAILABLE,
    }

    if not _mamba2_available:
        print("CRITICAL: mamba-ssm not installed. H100 Mamba performance will be TERRIBLE.")
        print("         Install: pip install mamba-ssm causal-conv1d --no-build-isolation")
    elif not _mamba2_optimized_kernels:
        print("WARNING: mamba-ssm Triton kernels not found. Performance may be suboptimal.")
    else:
        print("Mamba-2 optimized CUDA kernels: AVAILABLE")

    if FLASH_ATTN_AVAILABLE:
        print("FlashAttention-2: AVAILABLE (optimal for H100)")
    else:
        print("FlashAttention-2: Not available (using PyTorch SDPA fallback)")

    return status


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm (no mean subtraction, no bias).
    Required for Mamba stability at scale per Jamba findings.

    Formula: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


# =============================================================================
# Rotary Positional Embedding (RoPE)
# =============================================================================

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

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos/sin caches for efficiency."""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _extend_cache(self, seq_len: int):
        """Dynamically extend cache for longer sequences (length extrapolation)."""
        if seq_len > self.max_seq_len:
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
        required_len = offset + seq_len
        if required_len > self.max_seq_len:
            self._extend_cache(required_len)

        cos = self.cos_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = self.sin_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0).to(q.dtype)

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
        """
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)

        required_len = max(q_offset + seq_len_q, k_offset + seq_len_k)
        if required_len > self.max_seq_len:
            self._extend_cache(required_len)

        cos_q = self.cos_cache[q_offset : q_offset + seq_len_q].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin_q = self.sin_cache[q_offset : q_offset + seq_len_q].unsqueeze(0).unsqueeze(0).to(q.dtype)
        cos_k = self.cos_cache[k_offset : k_offset + seq_len_k].unsqueeze(0).unsqueeze(0).to(k.dtype)
        sin_k = self.sin_cache[k_offset : k_offset + seq_len_k].unsqueeze(0).unsqueeze(0).to(k.dtype)

        q_rot = (q * cos_q) + (self._rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (self._rotate_half(k) * sin_k)

        return q_rot, k_rot

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"


# =============================================================================
# Segment-Aware Flip (for BiMamba)
# =============================================================================

def segment_aware_flip(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flip sequences respecting document boundaries.

    CRITICAL: When processing packed sequences (multiple documents concatenated),
    we must flip WITHIN each document, not across document boundaries.

    Args:
        x: Input tensor
           - Padded mode: (batch, seq_len, d_model)
           - Packed mode: (total_tokens, d_model)
        cu_seqlens: Cumulative sequence lengths for packed mode
                   e.g., [0, 50, 80, 150] for 3 sequences

    Returns:
        Flipped tensor with same shape
    """
    if cu_seqlens is None:
        return torch.flip(x, dims=[1])

    batch_size = cu_seqlens.size(0) - 1
    flipped_segments = []

    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        segment = x[start:end]
        flipped_segment = torch.flip(segment, dims=[0])
        flipped_segments.append(flipped_segment)

    return torch.cat(flipped_segments, dim=0)


# =============================================================================
# SDPA Helper Functions
# =============================================================================

def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    training: bool = True,
) -> torch.Tensor:
    """
    PyTorch native scaled dot-product attention fallback.

    Args:
        q: Query tensor (batch, seq_len, n_heads, head_dim)
        k: Key tensor (batch, seq_len, n_heads, head_dim)
        v: Value tensor (batch, seq_len, n_heads, head_dim)
        dropout_p: Dropout probability
        causal: Whether to apply causal mask
        training: Whether in training mode

    Returns:
        Output tensor (batch, seq_len, n_heads, head_dim)
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=dropout_p if training else 0.0,
        is_causal=causal,
    )

    return out.transpose(1, 2)


def sdpa_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    PyTorch native scaled dot-product cross-attention fallback.
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )

    return out.transpose(1, 2)


# =============================================================================
# Bidirectional Attention (Encoder)
# =============================================================================

class BidirectionalAttention(nn.Module):
    """
    Bidirectional (non-causal) attention for encoder.
    Used for the 1:7 attention layers in the hybrid encoder.
    No KV cache needed since encoder processes full sequence at once.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        is_packed = cu_seqlens is not None

        if is_packed:
            residual = x
            x = self.norm(x)
            total_tokens = x.size(0)

            qkv = self.qkv_proj(x)
            qkv = qkv.view(total_tokens, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)

            if FLASH_ATTN_AVAILABLE:
                out = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                )
            else:
                batch_size = cu_seqlens.size(0) - 1
                outputs = []
                for i in range(batch_size):
                    start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                    q_i = q[start:end].unsqueeze(0)
                    k_i = k[start:end].unsqueeze(0)
                    v_i = v[start:end].unsqueeze(0)
                    out_i = sdpa_attention(q_i, k_i, v_i, self.dropout, causal=False, training=self.training)
                    outputs.append(out_i.squeeze(0))
                out = torch.cat(outputs, dim=0)

            out = out.view(total_tokens, self.d_model)
            out = self.out_proj(out)
            return residual + out
        else:
            residual = x
            x = self.norm(x)
            B, T, _ = x.shape

            qkv = self.qkv_proj(x)
            qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q, k = self.rope(q, k, offset=0)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.to(dtype=q.dtype)
                attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min

            if FLASH_ATTN_AVAILABLE and attention_mask is None:
                out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=False)
            else:
                q_sdpa = q.transpose(1, 2)
                k_sdpa = k.transpose(1, 2)
                v_sdpa = v.transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
                out = out.transpose(1, 2)

            out = out.view(B, T, self.d_model)
            out = self.out_proj(out)
            return residual + out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, head_dim={self.head_dim}, causal=False"


# =============================================================================
# Causal Self-Attention (Decoder)
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with RoPE and KV cache support.
    Used for the 1:7 attention layers in the hybrid decoder.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        is_packed = cu_seqlens is not None

        if is_packed:
            residual = x
            x = self.norm(x)
            total_tokens = x.size(0)

            qkv = self.qkv_proj(x)
            qkv = qkv.view(total_tokens, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)

            if FLASH_ATTN_AVAILABLE:
                out = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                )
            else:
                batch_size = cu_seqlens.size(0) - 1
                outputs = []
                for i in range(batch_size):
                    start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                    q_i = q[start:end].unsqueeze(0)
                    k_i = k[start:end].unsqueeze(0)
                    v_i = v[start:end].unsqueeze(0)
                    out_i = sdpa_attention(q_i, k_i, v_i, self.dropout, causal=True, training=self.training)
                    outputs.append(out_i.squeeze(0))
                out = torch.cat(outputs, dim=0)

            out = out.view(total_tokens, self.d_model)
            out = self.out_proj(out)
            return residual + out, (None, None)
        else:
            residual = x
            x = self.norm(x)
            B, T, _ = x.shape

            qkv = self.qkv_proj(x)
            qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q, k = self.rope(q, k, offset=offset)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            if kv_cache is not None:
                key_cache, value_cache = kv_cache
                if key_cache is not None:
                    k = torch.cat([key_cache, k], dim=1)
                    v = torch.cat([value_cache, v], dim=1)
            new_kv_cache = (k, v)

            if FLASH_ATTN_AVAILABLE:
                out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True)
            else:
                out = sdpa_attention(q, k, v, dropout_p=self.dropout, causal=True, training=self.training)

            out = out.view(B, T, self.d_model)
            out = self.out_proj(out)
            return residual + out, new_kv_cache

    def allocate_kv_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[None, None]:
        """Initialize empty KV cache."""
        return (None, None)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, head_dim={self.head_dim}, max_seq_len={self.max_seq_len}"


# =============================================================================
# Flash Cross-Attention (Decoder-to-Encoder)
# =============================================================================

class FlashCrossAttention(nn.Module):
    """
    Cross-attention using FlashAttention-2.
    Critical for fitting batch_size=64 with 8K sequences on H100.
    Uses O(N) memory instead of O(N^2).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        bias: bool = False,
        use_rope: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.kv_proj = nn.Linear(d_model, d_model * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        decoder_offset: int = 0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
        is_packed = cu_seqlens_q is not None

        if is_packed:
            residual = x
            x = self.norm(x)
            total_dec_tokens = x.size(0)
            total_enc_tokens = encoder_out.size(0)

            q = self.q_proj(x)
            q = q.view(total_dec_tokens, self.n_heads, self.head_dim)

            kv = self.kv_proj(encoder_out)
            kv = kv.view(total_enc_tokens, 2, self.n_heads, self.head_dim)
            k, v = kv.unbind(dim=1)

            if FLASH_ATTN_AVAILABLE:
                out = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                )
            else:
                batch_size = cu_seqlens_q.size(0) - 1
                outputs = []
                for i in range(batch_size):
                    q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
                    k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
                    q_i = q[q_start:q_end].unsqueeze(0)
                    k_i = k[k_start:k_end].unsqueeze(0)
                    v_i = v[k_start:k_end].unsqueeze(0)
                    out_i = sdpa_cross_attention(q_i, k_i, v_i, self.dropout, training=self.training)
                    outputs.append(out_i.squeeze(0))
                out = torch.cat(outputs, dim=0)

            out = out.view(total_dec_tokens, self.d_model)
            out = self.out_proj(out)
            return residual + out
        else:
            residual = x
            x = self.norm(x)
            B, T_dec, _ = x.shape
            _, T_enc, _ = encoder_out.shape

            q = self.q_proj(x)
            q = q.view(B, T_dec, self.n_heads, self.head_dim)

            kv = self.kv_proj(encoder_out)
            kv = kv.view(B, T_enc, 2, self.n_heads, self.head_dim)
            k, v = kv.unbind(dim=2)

            if self.rope is not None:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                q, k = self.rope.apply_to_qk(q, k, q_offset=decoder_offset, k_offset=0)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)

            if FLASH_ATTN_AVAILABLE:
                out = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=False)
            else:
                out = sdpa_cross_attention(q, k, v, dropout_p=self.dropout, training=self.training)

            out = out.view(B, T_dec, self.d_model)
            out = self.out_proj(out)
            return residual + out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, head_dim={self.head_dim}, use_rope={self.use_rope}"


# =============================================================================
# Mamba-2 Block Wrapper
# =============================================================================

class Mamba2BlockWrapper(nn.Module):
    """
    Wrapper around official Mamba2 with RMSNorm for stability.

    CRITICAL: Do NOT re-implement the SSD algorithm in PyTorch.
    The official CUDA kernels are 10-50x faster.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if not _mamba2_available:
            raise ImportError(
                "mamba-ssm is required for Mamba2BlockWrapper. "
                "Install with: pip install mamba-ssm (requires CUDA on Linux)"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.layer_idx = layer_idx

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        if inference_params is not None:
            conv_state, ssm_state = inference_params
            x, conv_state_out, ssm_state_out = self.mamba.step(x, conv_state, ssm_state)
            conv_state.copy_(conv_state_out)
            ssm_state.copy_(ssm_state_out)
        else:
            x = self.mamba(x)

        return residual + x

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate inference cache for autoregressive decoding."""
        d_inner = self.d_model * self.expand
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        conv_state = torch.zeros(batch_size, d_inner, self.d_conv, dtype=dtype, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.d_state, dtype=dtype, device=device)

        return conv_state, ssm_state

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_state={self.d_state}, d_conv={self.d_conv}, expand={self.expand}"


# =============================================================================
# Bidirectional Mamba Block (Encoder)
# =============================================================================

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba for Encoder.

    CRITICAL: Concatenate OUTPUTS (y), not internal states (h)!

    Process:
    1. Forward scan: y_fwd = Mamba(x) on FULL d_model
    2. Backward scan: y_bwd = Flip(Mamba(Flip(x))) on FULL d_model
    3. Concatenate: [y_fwd; y_bwd] gives (B, L, 2*d_model)
    4. Project: Linear(2*d_model -> d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if not _mamba2_available:
            raise ImportError(
                "mamba-ssm is required for BiMambaBlock. "
                "Install with: pip install mamba-ssm (requires CUDA on Linux)"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, **factory_kwargs)
        self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, **factory_kwargs)
        self.out_proj = nn.Linear(d_model * 2, d_model, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        y_fwd = self.mamba_fwd(x)

        x_flipped = segment_aware_flip(x, cu_seqlens)
        y_bwd_rev = self.mamba_bwd(x_flipped)
        y_bwd = segment_aware_flip(y_bwd_rev, cu_seqlens)

        out = torch.cat([y_fwd, y_bwd], dim=-1)
        out = self.out_proj(out)

        return residual + out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_state={self.d_state}, d_conv={self.d_conv}, expand={self.expand}, bidirectional=True"
