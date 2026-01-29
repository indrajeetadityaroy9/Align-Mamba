"""Fused Triton kernels: RMSNorm and label-smoothing cross-entropy."""

from .rmsnorm import fused_rmsnorm
from .loss import fused_cross_entropy_loss

__all__ = ["fused_rmsnorm", "fused_cross_entropy_loss"]
