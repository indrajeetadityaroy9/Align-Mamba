"""
Mamba module - Mamba-2 SSM blocks for the hybrid architecture.

Provides:
- Mamba2BlockWrapper: Wrapped Mamba2 with RMSNorm
- BiMambaBlock: Bidirectional Mamba for encoder
- segment_aware_flip: Document-boundary-respecting sequence flip
"""

from .wrapper import Mamba2BlockWrapper
from .bimamba import BiMambaBlock, segment_aware_flip

__all__ = [
    "Mamba2BlockWrapper",
    "BiMambaBlock",
    "segment_aware_flip",
]
