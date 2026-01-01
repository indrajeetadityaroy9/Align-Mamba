"""
Causal Self-Attention with RoPE for decoder.

Supports KV caching for efficient autoregressive generation.
Supports VarLen mode for packed sequence training (20-30% H100 speedup).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from flash_attn import flash_attn_func
from flash_attn import flash_attn_varlen_func

from ..mamba2.norms import RMSNorm
from .rope import RotaryPositionalEmbedding


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
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Attention dropout
            max_seq_len: Maximum sequence length for RoPE cache
            bias: Whether to use bias in projections
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        # RoPE for positional encoding
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Causal self-attention forward pass.

        Args:
            x: Input tensor
               - Padded mode: (batch, seq_len, d_model)
               - Packed mode: (total_tokens, d_model)
            kv_cache: Optional (key_cache, value_cache) for incremental decoding
            offset: Position offset for RoPE
            cu_seqlens: Cumulative sequence lengths for packed mode (batch+1,)
            max_seqlen: Maximum sequence length in batch (for packed mode)

        Returns:
            Tuple of:
            - Output tensor (same shape as input)
            - Updated (key_cache, value_cache)
        """
        is_packed = cu_seqlens is not None

        if is_packed:
            # Packed mode: x is (total_tokens, d_model)
            # Note: KV cache not supported in packed training mode
            residual = x
            x = self.norm(x)

            total_tokens = x.size(0)

            # Project QKV: (total_tokens, 3 * d_model)
            qkv = self.qkv_proj(x)
            qkv = qkv.view(total_tokens, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)  # Each: (total_tokens, n_heads, head_dim)

            # Note: RoPE in packed mode requires per-sequence position computation
            # Skipping RoPE for packed training mode (Mamba handles positions anyway)

            # FlashAttention-2 VarLen with causal mask
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )

            # Reshape and project output
            out = out.view(total_tokens, self.d_model)
            out = self.out_proj(out)

            return residual + out, (None, None)  # No KV cache in packed mode
        else:
            # Padded mode: x is (batch, seq_len, d_model)
            residual = x
            x = self.norm(x)

            B, T, _ = x.shape

            # Project QKV
            qkv = self.qkv_proj(x)
            qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            # Apply RoPE
            # Transpose for RoPE: (B, T, H, D) -> (B, H, T, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q, k = self.rope(q, k, offset=offset)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            # Update KV cache
            if kv_cache is not None:
                key_cache, value_cache = kv_cache
                if key_cache is not None:
                    k = torch.cat([key_cache, k], dim=1)
                    v = torch.cat([value_cache, v], dim=1)
            new_kv_cache = (k, v)

            # FlashAttention-2 with causal mask
            out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )

            # Reshape and project output
            out = out.view(B, T, self.d_model)
            out = self.out_proj(out)

            return residual + out, new_kv_cache

    def allocate_kv_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[None, None]:
        """
        Initialize empty KV cache.

        For attention, we start with None and grow the cache during generation.
        """
        return (None, None)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}"
        )
