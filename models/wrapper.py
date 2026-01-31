"""Polarized Mamba-2 Block - SOTA SSM with recency bias mitigation.

Polarization addresses exponential decay in SSMs where distant tokens are
"under-reaching and forgotten rapidly" (Theorem 3.1, arXiv:2501.00658).

Three channels:
- Learnable: Standard Mamba with adaptive A matrix
- Zero (A=0): No memory, pure local processing
- One (A=1): Perfect memory via cumulative sum

Reference: arXiv:2501.00658 (ICLR 2025)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from mamba_ssm import Mamba2

from .normalization import RMSNorm
from .utils import process_segments_unidirectional


class PolarizedMamba2Block(nn.Module):
    """Mamba2 with polarized channels for recency bias mitigation."""

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

        # Learnable channel (standard Mamba)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

        # Polarized channels: A=0 (local) + A=1 (global cumsum)
        self.zero_proj = nn.Linear(d_model, d_inner, bias=False, **factory_kwargs)
        self.one_proj = nn.Linear(d_model, d_inner, bias=False, **factory_kwargs)
        self.fusion = nn.Linear(d_inner * 3, d_model, bias=False, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.contiguous()

        # Learnable channel
        if inference_params is not None:
            conv_state, ssm_state = inference_params
            mamba_out, conv_state_out, ssm_state_out = self.mamba.step(x, conv_state, ssm_state)
            conv_state.copy_(conv_state_out)
            ssm_state.copy_(ssm_state_out)
        elif cu_seqlens is not None:
            mamba_out = process_segments_unidirectional(x, self.mamba, cu_seqlens)
        else:
            mamba_out = self.mamba(x)

        # Polarized channels
        y_zero = self.zero_proj(x)  # A=0: no temporal dependency
        y_one = torch.cumsum(self.one_proj(x), dim=1)  # A=1: perfect memory

        # Fuse all channels
        fused = torch.cat([mamba_out, y_zero, y_one], dim=-1)
        out = self.fusion(fused)

        return residual + out

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_inner = self.d_model * self.expand
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        conv_state = torch.zeros(batch_size, d_inner, self.d_conv, dtype=dtype, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.d_state, dtype=dtype, device=device)

        return conv_state, ssm_state
