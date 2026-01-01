"""
Bidirectional Mamba Block for Encoder.

CRITICAL: Mamba is natively causal (left-to-right).
Simply removing causal masking does NOT make it bidirectional - it breaks.

Solution: Run forward scan + backward scan, concatenate outputs.
This gives true bidirectional context for the encoder.
"""

import torch
import torch.nn as nn
from typing import Optional

from mamba_ssm import Mamba2

from .norms import RMSNorm


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba for Encoder.

    Runs Mamba twice:
    1. Forward scan (left-to-right) on first half of dimensions
    2. Backward scan (right-to-left) on second half of dimensions

    Without this, the Encoder is just a weak LM that can't see sentence endings.
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
            d_model: Model dimension (must be even for split)
            d_state: SSM state dimension
            d_conv: Local convolution kernel size
            expand: Block expansion factor
            headdim: Head dimension
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % 2 == 0, "d_model must be even for BiMamba"

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Forward direction Mamba (processes d_model // 2 dimensions)
        self.mamba_fwd = Mamba2(
            d_model=d_model // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

        # Backward direction Mamba (processes d_model // 2 dimensions)
        self.mamba_bwd = Mamba2(
            d_model=d_model // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

        # Output projection to fuse forward and backward
        self.out_proj = nn.Linear(d_model, d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split input into two halves
        x_fwd, x_bwd = x.chunk(2, dim=-1)

        # Forward scan (left-to-right)
        out_fwd = self.mamba_fwd(x_fwd)

        # Backward scan (right-to-left)
        # Flip sequence, process, flip back
        x_bwd_rev = torch.flip(x_bwd, dims=[1])
        out_bwd_rev = self.mamba_bwd(x_bwd_rev)
        out_bwd = torch.flip(out_bwd_rev, dims=[1])

        # Concatenate forward and backward outputs
        out = torch.cat([out_fwd, out_bwd], dim=-1)

        # Project back to d_model
        out = self.out_proj(out)

        return residual + out

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, expand={self.expand}, bidirectional=True"
        )
