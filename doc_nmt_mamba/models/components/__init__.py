"""
Components module - Generic building blocks for the hybrid architecture.

Provides:
- RMSNorm: Stable normalization for 200M+ params
- RotaryPositionalEmbedding: RoPE for attention layers only
- Attention layers: Bidirectional, Causal, Cross-attention
"""

from .normalization import (
    RMSNorm,
    FLASH_ATTN_AVAILABLE,
    check_h100_kernel_status,
)
from .embeddings import RotaryPositionalEmbedding
from .attention import (
    sdpa_attention,
    sdpa_cross_attention,
    BidirectionalAttention,
    CausalSelfAttention,
    FlashCrossAttention,
)

__all__ = [
    # Normalization
    "RMSNorm",
    "FLASH_ATTN_AVAILABLE",
    "check_h100_kernel_status",
    # Embeddings
    "RotaryPositionalEmbedding",
    # Attention
    "sdpa_attention",
    "sdpa_cross_attention",
    "BidirectionalAttention",
    "CausalSelfAttention",
    "FlashCrossAttention",
]
