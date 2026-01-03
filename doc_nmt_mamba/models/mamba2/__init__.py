"""Mamba-2 block implementations."""

import warnings

# Always available (pure PyTorch)
from .norms import RMSNorm
from .bimamba import segment_aware_flip

# CUDA-dependent components
_mamba2_available = False

try:
    from .mamba2_wrapper import Mamba2BlockWrapper
    from .bimamba import BiMambaBlock
    _mamba2_available = True
except ImportError as e:
    warnings.warn(f"Mamba2 components not available: {e}")

    class _CUDARequired:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "This module requires mamba-ssm (CUDA only). "
                "Install with: pip install mamba-ssm"
            )

    Mamba2BlockWrapper = _CUDARequired
    BiMambaBlock = _CUDARequired


__all__ = [
    "RMSNorm",
    "Mamba2BlockWrapper",
    "BiMambaBlock",
    "segment_aware_flip",
]
