"""Polarized Mamba block with A=0/A=1 channels (arXiv:2501.00658)."""

from typing import Optional

import torch
import torch.nn as nn

from mamba_ssm import Mamba2

from align_mamba.models.components import RMSNorm
from align_mamba.models.registry import BlockRegistry


@BlockRegistry.register("polarized")
class PolarizedMamba2Block(nn.Module):
    """Mamba with A=0/A=1 channels for recency bias mitigation (arXiv:2501.00658)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        polarized_channels: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,  # Accept extra kwargs for registry compatibility
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.polarized_channels = polarized_channels
        d_inner = d_model * expand

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

        self.zero_proj = nn.Linear(d_model, d_inner, bias=False, **factory_kwargs)
        self.one_proj = nn.Linear(d_model, d_inner, bias=False, **factory_kwargs)
        self.fusion = nn.Linear(d_inner * 3, d_model, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.contiguous()
        mamba_out = self.mamba(x)

        y_zero = self.zero_proj(x)
        y_one = torch.cumsum(self.one_proj(x), dim=1)

        fused = torch.cat([mamba_out, y_zero, y_one], dim=-1)
        out = self.fusion(fused)

        return residual + out
