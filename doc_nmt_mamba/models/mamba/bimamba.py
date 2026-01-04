"""
Bidirectional Mamba Block for Encoder.

Contains:
- segment_aware_flip: Flip sequences respecting document boundaries
- BiMambaBlock: Bidirectional Mamba with forward + backward scans
"""

import torch
import torch.nn as nn
from typing import Optional

from ..components.normalization import RMSNorm

# Import Mamba2 conditionally
_mamba2_available = False
Mamba2 = None

try:
    from mamba_ssm import Mamba2 as _Mamba2
    Mamba2 = _Mamba2
    _mamba2_available = True
except ImportError:
    pass


# =============================================================================
# Segment-Aware Flip (for BiMamba)
# =============================================================================

def segment_aware_flip(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flip sequences respecting document boundaries.

    CRITICAL: When processing packed sequences (multiple documents concatenated),
    we must flip WITHIN each document, not across document boundaries.

    Args:
        x: Input tensor
           - Padded mode: (batch, seq_len, d_model)
           - Packed mode: (total_tokens, d_model)
        cu_seqlens: Cumulative sequence lengths for packed mode
                   e.g., [0, 50, 80, 150] for 3 sequences

    Returns:
        Flipped tensor with same shape
    """
    if cu_seqlens is None:
        return torch.flip(x, dims=[1])

    batch_size = cu_seqlens.size(0) - 1
    flipped_segments = []

    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        segment = x[start:end]
        flipped_segment = torch.flip(segment, dims=[0])
        flipped_segments.append(flipped_segment)

    return torch.cat(flipped_segments, dim=0)


# =============================================================================
# Bidirectional Mamba Block (Encoder)
# =============================================================================

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba for Encoder.

    CRITICAL: Concatenate OUTPUTS (y), not internal states (h)!

    Process:
    1. Forward scan: y_fwd = Mamba(x) on FULL d_model
    2. Backward scan: y_bwd = Flip(Mamba(Flip(x))) on FULL d_model
    3. Concatenate: [y_fwd; y_bwd] gives (B, L, 2*d_model)
    4. Project: Linear(2*d_model -> d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if not _mamba2_available:
            raise ImportError(
                "mamba-ssm is required for BiMambaBlock. "
                "Install with: pip install mamba-ssm (requires CUDA on Linux)"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, **factory_kwargs)
        self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, **factory_kwargs)
        self.out_proj = nn.Linear(d_model * 2, d_model, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        y_fwd = self.mamba_fwd(x)

        x_flipped = segment_aware_flip(x, cu_seqlens)
        y_bwd_rev = self.mamba_bwd(x_flipped)
        y_bwd = segment_aware_flip(y_bwd_rev, cu_seqlens)

        out = torch.cat([y_fwd, y_bwd], dim=-1)
        out = self.out_proj(out)

        return residual + out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_state={self.d_state}, d_conv={self.d_conv}, expand={self.expand}, bidirectional=True"
