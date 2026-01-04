"""
Proxy file for backward compatibility.

This file re-exports all hybrid architecture components from their new modular locations.
For new code, import directly from:
- doc_nmt_mamba.models.align_mamba
- doc_nmt_mamba.models.encoder_decoder
"""

# Layer types and utilities
from .align_mamba import (
    LayerType,
    count_layer_types,
    MambaState,
    AttentionKVCache,
    HybridCacheParams,
    HybridBlock,
    HybridBiMambaEncoder,
    HybridMambaDecoder,
)

# Model configuration and full model
from .encoder_decoder import (
    ModelConfig,
    HybridMambaEncoderDecoder,
)

__all__ = [
    # Layer types
    "LayerType",
    "count_layer_types",
    # Inference state
    "MambaState",
    "AttentionKVCache",
    "HybridCacheParams",
    # Architecture components
    "HybridBlock",
    "HybridBiMambaEncoder",
    "HybridMambaDecoder",
    # Configuration and full model
    "ModelConfig",
    "HybridMambaEncoderDecoder",
]
