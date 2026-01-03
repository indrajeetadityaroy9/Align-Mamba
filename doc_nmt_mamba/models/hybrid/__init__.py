"""Hybrid Mamba-Attention architecture."""

import warnings

# These are always available (pure Python/PyTorch)
from .layer_builder import (
    LayerType,
    compute_attention_positions,
    compute_hybrid_positions,
    count_layer_types,
)
from .decoder import MambaState, AttentionKVCache

# CUDA-dependent components
_hybrid_available = False

try:
    from .layer_builder import (
        build_encoder_layers,
        build_decoder_layers,
        HybridBlock,
    )
    from .encoder import HybridBiMambaEncoder
    from .decoder import HybridMambaDecoder

    _hybrid_available = True

except ImportError as e:
    warnings.warn(f"Hybrid model components not available: {e}")

    # Placeholder classes
    class _CUDARequired:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "This module requires mamba-ssm (CUDA only). "
                "Install with: pip install mamba-ssm"
            )

    HybridBlock = _CUDARequired
    HybridBiMambaEncoder = _CUDARequired
    HybridMambaDecoder = _CUDARequired

    def build_encoder_layers(*args, **kwargs):
        raise ImportError("mamba-ssm required for build_encoder_layers")

    def build_decoder_layers(*args, **kwargs):
        raise ImportError("mamba-ssm required for build_decoder_layers")


__all__ = [
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
]
