"""
Attention Mechanisms for Hybrid Mamba-Attention Architecture.

Contains:
- SDPA helper functions (PyTorch native fallback)
- BidirectionalAttention: Non-causal attention for encoder
- CausalSelfAttention: Causal self-attention for decoder
- FlashCrossAttention: Cross-attention using FlashAttention-2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .normalization import FLASH_ATTN_AVAILABLE, RMSNorm
from .embeddings import RotaryPositionalEmbedding

# Import flash attention functions if available
if FLASH_ATTN_AVAILABLE:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
else:
    flash_attn_func = None
    flash_attn_varlen_func = None


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
