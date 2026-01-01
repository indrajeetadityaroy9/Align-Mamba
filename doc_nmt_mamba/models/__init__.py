"""
Document-Level NMT Models with Hybrid Mamba-Attention Architecture.

This package provides:
- Mamba-2 blocks (causal and bidirectional)
- Attention modules (RoPE, FlashAttention-based)
- Hybrid encoder-decoder for document-level NMT
"""

from .encoder_decoder import (
    ModelConfig,
    HybridCacheParams,
    HybridMambaEncoderDecoder,
)

from .mamba2 import (
    RMSNorm,
    Mamba2BlockWrapper,
    BiMambaBlock,
)

from .attention import (
    RotaryPositionalEmbedding,
    FlashCrossAttention,
    CausalSelfAttention,
    BidirectionalAttention,
)

from .hybrid import (
    LayerType,
    compute_attention_positions,
    build_encoder_layers,
    build_decoder_layers,
    count_layer_types,
    HybridBiMambaEncoder,
    HybridMambaDecoder,
    MambaState,
    AttentionKVCache,
)

__all__ = [
    # Main model
    "ModelConfig",
    "HybridCacheParams",
    "HybridMambaEncoderDecoder",
    # Mamba blocks
    "RMSNorm",
    "Mamba2BlockWrapper",
    "BiMambaBlock",
    # Attention
    "RotaryPositionalEmbedding",
    "FlashCrossAttention",
    "CausalSelfAttention",
    "BidirectionalAttention",
    # Hybrid architecture
    "LayerType",
    "compute_attention_positions",
    "build_encoder_layers",
    "build_decoder_layers",
    "count_layer_types",
    "HybridBiMambaEncoder",
    "HybridMambaDecoder",
    "MambaState",
    "AttentionKVCache",
]
