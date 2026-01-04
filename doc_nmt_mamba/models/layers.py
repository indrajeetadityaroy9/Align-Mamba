"""
Proxy file for backward compatibility.

This file re-exports all layer components from their new modular locations.
For new code, import directly from:
- doc_nmt_mamba.models.components.normalization
- doc_nmt_mamba.models.components.embeddings
- doc_nmt_mamba.models.components.attention
- doc_nmt_mamba.models.mamba.bimamba
- doc_nmt_mamba.models.mamba.wrapper
"""

# Normalization and CUDA checks
from .components.normalization import (
    RMSNorm,
    FLASH_ATTN_AVAILABLE,
    check_h100_kernel_status,
)

# Positional embeddings
from .components.embeddings import RotaryPositionalEmbedding

# Attention mechanisms
from .components.attention import (
    sdpa_attention,
    sdpa_cross_attention,
    BidirectionalAttention,
    CausalSelfAttention,
    FlashCrossAttention,
)

# Mamba blocks
from .mamba.bimamba import segment_aware_flip, BiMambaBlock
from .mamba.wrapper import Mamba2BlockWrapper

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
    # Mamba
    "segment_aware_flip",
    "BiMambaBlock",
    "Mamba2BlockWrapper",
]
