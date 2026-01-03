"""
Document-Level NMT Models with Hybrid Mamba-Attention Architecture.

This package provides:
- Mamba-2 blocks (causal and bidirectional)
- Attention modules (RoPE, FlashAttention-based)
- Hybrid encoder-decoder for document-level NMT

Note: Full model functionality requires mamba-ssm (CUDA only).
Some components (attention, segment_aware_flip) work without CUDA.
"""

import warnings

# Always available: attention modules
from .attention import (
    RotaryPositionalEmbedding,
    FlashCrossAttention,
    CausalSelfAttention,
    BidirectionalAttention,
)

# Always available: pure PyTorch utilities
from .mamba2.norms import RMSNorm
from .mamba2.bimamba import segment_aware_flip

# Always available: checkpoint utilities
from .checkpoint_utils import (
    load_model_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

# Conditionally available: CUDA-dependent modules
_cuda_modules_available = False

try:
    from .encoder_decoder import (
        ModelConfig,
        HybridCacheParams,
        HybridMambaEncoderDecoder,
    )

    from .mamba2 import (
        Mamba2BlockWrapper,
        BiMambaBlock,
    )

    from .hybrid import (
        LayerType,
        compute_attention_positions,
        compute_hybrid_positions,
        build_encoder_layers,
        build_decoder_layers,
        count_layer_types,
        HybridBlock,
        HybridBiMambaEncoder,
        HybridMambaDecoder,
        MambaState,
        AttentionKVCache,
    )

    _cuda_modules_available = True

except ImportError as e:
    warnings.warn(
        f"CUDA-dependent modules not available: {e}. "
        "Install mamba-ssm for full functionality."
    )

    # Provide placeholder classes that raise helpful errors
    class _CUDARequired:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "This module requires mamba-ssm (CUDA only). "
                "Install with: pip install mamba-ssm"
            )

    ModelConfig = _CUDARequired
    HybridCacheParams = _CUDARequired
    HybridMambaEncoderDecoder = _CUDARequired
    Mamba2BlockWrapper = _CUDARequired
    BiMambaBlock = _CUDARequired
    HybridBlock = _CUDARequired
    HybridBiMambaEncoder = _CUDARequired
    HybridMambaDecoder = _CUDARequired

    # These are still importable for type checking
    from .hybrid.layer_builder import LayerType
    from .hybrid.layer_builder import (
        compute_attention_positions,
        compute_hybrid_positions,
        count_layer_types,
    )
    from .hybrid.decoder import MambaState, AttentionKVCache

    # Dummy builder functions
    def build_encoder_layers(*args, **kwargs):
        raise ImportError("mamba-ssm required for build_encoder_layers")

    def build_decoder_layers(*args, **kwargs):
        raise ImportError("mamba-ssm required for build_decoder_layers")


__all__ = [
    # Main model
    "ModelConfig",
    "HybridCacheParams",
    "HybridMambaEncoderDecoder",
    # Mamba blocks
    "RMSNorm",
    "Mamba2BlockWrapper",
    "BiMambaBlock",
    "segment_aware_flip",
    # Attention
    "RotaryPositionalEmbedding",
    "FlashCrossAttention",
    "CausalSelfAttention",
    "BidirectionalAttention",
    # Hybrid architecture
    "LayerType",
    "compute_attention_positions",
    "compute_hybrid_positions",
    "build_encoder_layers",
    "build_decoder_layers",
    "count_layer_types",
    "HybridBlock",
    "HybridBiMambaEncoder",
    "HybridMambaDecoder",
    "MambaState",
    "AttentionKVCache",
    # Checkpoint utilities
    "load_model_from_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
    # Availability flag
    "_cuda_modules_available",
]
