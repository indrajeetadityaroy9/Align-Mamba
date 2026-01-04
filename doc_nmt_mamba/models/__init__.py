"""
Document-Level NMT Models with Hybrid Mamba-Attention Architecture.

This package provides:
- Building blocks: RMSNorm, RoPE, Attention layers, Mamba blocks
- Hybrid architecture: Encoder, Decoder, Full Model
- Checkpoint utilities: save/load with embedded config

Modular structure (for reviewer readability):
- components/: Generic building blocks (normalization, embeddings, attention)
- mamba/: Mamba-2 SSM blocks (wrapper, bimamba)
- align_mamba.py: NOVEL contributions (HybridBlock, Encoder, Decoder)
- encoder_decoder.py: ModelConfig, full EncoderDecoder wrapper
- checkpoints.py: Checkpoint save/load utilities

Backward compatibility:
- layers.py: Proxy file re-exporting from components/ and mamba/
- modeling_hybrid.py: Proxy file re-exporting from align_mamba and encoder_decoder

Note: Full model functionality requires mamba-ssm (CUDA only).
Some components (attention, segment_aware_flip) work without CUDA.
"""

import warnings

# Always available: building blocks from layers.py (proxy file)
from .layers import (
    RMSNorm,
    RotaryPositionalEmbedding,
    segment_aware_flip,
    sdpa_attention,
    sdpa_cross_attention,
    BidirectionalAttention,
    CausalSelfAttention,
    FlashCrossAttention,
    FLASH_ATTN_AVAILABLE,
)

# Always available: checkpoint utilities (renamed from cache_utils.py)
from .checkpoints import (
    load_model_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

# Conditionally available: CUDA-dependent modules
_cuda_modules_available = False

try:
    # Mamba blocks (require mamba-ssm)
    from .layers import (
        Mamba2BlockWrapper,
        BiMambaBlock,
    )

    # Architecture classes (from proxy file)
    from .modeling_hybrid import (
        LayerType,
        count_layer_types,
        MambaState,
        AttentionKVCache,
        HybridCacheParams,
        ModelConfig,
        HybridBlock,
        HybridBiMambaEncoder,
        HybridMambaDecoder,
        HybridMambaEncoderDecoder,
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

    Mamba2BlockWrapper = _CUDARequired
    BiMambaBlock = _CUDARequired
    ModelConfig = _CUDARequired
    HybridCacheParams = _CUDARequired
    HybridMambaEncoderDecoder = _CUDARequired
    HybridBlock = _CUDARequired
    HybridBiMambaEncoder = _CUDARequired
    HybridMambaDecoder = _CUDARequired

    # These are still importable for type checking
    from .align_mamba import (
        LayerType,
        count_layer_types,
        MambaState,
        AttentionKVCache,
    )


__all__ = [
    # Building blocks - always available
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "segment_aware_flip",
    "sdpa_attention",
    "sdpa_cross_attention",
    "BidirectionalAttention",
    "CausalSelfAttention",
    "FlashCrossAttention",
    "FLASH_ATTN_AVAILABLE",
    # Mamba blocks - require CUDA
    "Mamba2BlockWrapper",
    "BiMambaBlock",
    # Architecture
    "LayerType",
    "count_layer_types",
    "MambaState",
    "AttentionKVCache",
    "HybridCacheParams",
    "ModelConfig",
    "HybridBlock",
    "HybridBiMambaEncoder",
    "HybridMambaDecoder",
    "HybridMambaEncoderDecoder",
    # Checkpoint utilities
    "load_model_from_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
    # Availability flag
    "_cuda_modules_available",
]
