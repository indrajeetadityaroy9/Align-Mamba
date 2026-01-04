"""
Normalization and CUDA kernel utilities.

Contains:
- RMSNorm: Root Mean Square Layer Normalization
- CUDA kernel availability checks for H100 optimization
"""

import torch
import torch.nn as nn
import warnings

# =============================================================================
# CUDA-dependent imports with fallbacks
# H100 Kernel Selection: Prefer Dao-AI-Lab CUDA kernels for 10-50x speedup
# =============================================================================

# Mamba-2 (CUDA only - CRITICAL for H100 performance)
_mamba2_available = False
_mamba2_optimized_kernels = False
Mamba2 = None

try:
    from mamba_ssm import Mamba2 as _Mamba2
    Mamba2 = _Mamba2
    _mamba2_available = True

    # Check for optimized Triton kernels (critical for H100 performance)
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        _mamba2_optimized_kernels = True
    except ImportError:
        _mamba2_optimized_kernels = False
        warnings.warn(
            "mamba-ssm Triton kernels not available. "
            "Performance on H100 will be suboptimal. "
            "Reinstall with: pip install mamba-ssm --no-build-isolation"
        )
except ImportError:
    warnings.warn(
        "CRITICAL: mamba-ssm not available. Mamba2BlockWrapper and BiMambaBlock will not work. "
        "On H100, this means 10-50x slower training! "
        "Install with: pip install mamba-ssm causal-conv1d --no-build-isolation (requires CUDA)"
    )

# FlashAttention-2 (optional, falls back to PyTorch SDPA)
# H100 has native SDPA support, but FlashAttention-2 is still faster
FLASH_ATTN_AVAILABLE = False
flash_attn_func = None
flash_attn_varlen_func = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    # PyTorch SDPA fallback is acceptable on H100 (native support)
    pass


def check_h100_kernel_status():
    """
    Check and report H100 kernel optimization status.

    Call this at training start to ensure optimal configuration.
    """
    status = {
        "mamba2_available": _mamba2_available,
        "mamba2_optimized": _mamba2_optimized_kernels,
        "flash_attn_available": FLASH_ATTN_AVAILABLE,
    }

    if not _mamba2_available:
        print("CRITICAL: mamba-ssm not installed. H100 Mamba performance will be TERRIBLE.")
        print("         Install: pip install mamba-ssm causal-conv1d --no-build-isolation")
    elif not _mamba2_optimized_kernels:
        print("WARNING: mamba-ssm Triton kernels not found. Performance may be suboptimal.")
    else:
        print("Mamba-2 optimized CUDA kernels: AVAILABLE")

    if FLASH_ATTN_AVAILABLE:
        print("FlashAttention-2: AVAILABLE (optimal for H100)")
    else:
        print("FlashAttention-2: Not available (using PyTorch SDPA fallback)")

    return status


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm (no mean subtraction, no bias).
    Required for Mamba stability at scale per Jamba findings.

    Formula: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
