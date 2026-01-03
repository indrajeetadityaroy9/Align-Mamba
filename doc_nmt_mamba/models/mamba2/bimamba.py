"""
Bidirectional Mamba Block for Encoder.

CRITICAL: Mamba is natively causal (left-to-right).
Simply removing causal masking does NOT make it bidirectional - it breaks.

Solution: Run forward scan + backward scan, concatenate OUTPUTS (not states).
This gives true bidirectional context for the encoder.

From the plan:
    y_fwd = Mamba(x)                    # (B, L, D)
    y_bwd = Flip(Mamba(Flip(x)))        # Reverse, process, reverse back
    out = Linear(Concat(y_fwd, y_bwd))  # (B, L, 2D) → (B, L, D)

CAUTION for packed sequences:
    Flip must respect document boundaries (cu_seqlens)!
    Do NOT flip across <doc> separators.
"""

import torch
import torch.nn as nn
from typing import Optional, List
import warnings

# Lazy import for mamba_ssm (CUDA-only)
_mamba2_available = False
Mamba2 = None

try:
    from mamba_ssm import Mamba2 as _Mamba2
    Mamba2 = _Mamba2
    _mamba2_available = True
except ImportError:
    pass  # Will be handled in BiMambaBlock.__init__

from .norms import RMSNorm


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
        # Padded mode: simple flip along sequence dimension
        return torch.flip(x, dims=[1])

    # Packed mode: flip each segment separately
    # cu_seqlens: [0, len1, len1+len2, ...]
    batch_size = cu_seqlens.size(0) - 1
    flipped_segments = []

    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        segment = x[start:end]
        # Flip this segment (along dim=0 since it's (seq_len, d_model))
        flipped_segment = torch.flip(segment, dims=[0])
        flipped_segments.append(flipped_segment)

    return torch.cat(flipped_segments, dim=0)


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba for Encoder.

    CRITICAL from plan: Concatenate OUTPUTS (y), not internal states (h)!

    Process:
    1. Forward scan: y_fwd = Mamba(x) on FULL d_model
    2. Backward scan: y_bwd = Flip(Mamba(Flip(x))) on FULL d_model
    3. Concatenate: [y_fwd; y_bwd] gives (B, L, 2*d_model)
    4. Project: Linear(2*d_model → d_model)

    This differs from the "split input" approach - we process the FULL
    input in both directions for richer bidirectional context.
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
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution kernel size
            expand: Block expansion factor
            headdim: Head dimension
            device: Device for parameters
            dtype: Data type for parameters
        """
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

        # Forward direction Mamba (processes FULL d_model)
        self.mamba_fwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

        # Backward direction Mamba (processes FULL d_model)
        # Separate weights for backward scan
        self.mamba_bwd = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

        # Output projection to fuse forward and backward (2*d_model → d_model)
        self.out_proj = nn.Linear(d_model * 2, d_model, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Bidirectional forward pass.

        Args:
            x: Input tensor
               - Padded mode: (batch, seq_len, d_model)
               - Packed mode: (total_tokens, d_model) with cu_seqlens
            cu_seqlens: Cumulative sequence lengths for packed mode

        Returns:
            Output tensor (same shape as input)
        """
        residual = x
        x = self.norm(x)

        # Forward scan (left-to-right) on full input
        y_fwd = self.mamba_fwd(x)

        # Backward scan (right-to-left)
        # CRITICAL: Use segment-aware flip for packed sequences
        x_flipped = segment_aware_flip(x, cu_seqlens)
        y_bwd_rev = self.mamba_bwd(x_flipped)
        y_bwd = segment_aware_flip(y_bwd_rev, cu_seqlens)

        # Concatenate forward and backward outputs
        # (B, L, D) + (B, L, D) → (B, L, 2D)
        out = torch.cat([y_fwd, y_bwd], dim=-1)

        # Project back to d_model
        out = self.out_proj(out)

        return residual + out

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, expand={self.expand}, bidirectional=True, "
            f"output_concat=True"
        )
