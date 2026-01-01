"""
Mamba-2 Block Wrapper around official mamba-ssm library.

CRITICAL: Do NOT re-implement the SSD algorithm in PyTorch.
The official CUDA kernels are 10-50x faster.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from mamba_ssm import Mamba2

from .norms import RMSNorm


class Mamba2BlockWrapper(nn.Module):
    """
    Wrapper around official Mamba2 with RMSNorm for stability.

    The actual SSM computation uses optimized CUDA kernels from mamba-ssm.
    This wrapper adds:
    - Pre-normalization with RMSNorm
    - Residual connection
    - Layer index tracking for inference state management
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension (64-128 typical)
            d_conv: Local convolution kernel size
            expand: Block expansion factor
            headdim: Head dimension for grouped attention pattern
            layer_idx: Layer index for inference cache management
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.layer_idx = layer_idx

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Use official Mamba2 from mamba-ssm
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            inference_params: Optional (conv_state, ssm_state) for autoregressive decoding.
                             States are updated IN-PLACE during step().

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        if inference_params is not None:
            # Autoregressive mode - use step() for single token
            # mamba.step() updates states in-place and returns output
            conv_state, ssm_state = inference_params
            x, conv_state_out, ssm_state_out = self.mamba.step(x, conv_state, ssm_state)
            # Update states in-place for next step
            conv_state.copy_(conv_state_out)
            ssm_state.copy_(ssm_state_out)
        else:
            # Training mode - parallel computation
            x = self.mamba(x)

        return residual + x

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate inference cache tensors for autoregressive decoding.

        Returns:
            Tuple of (conv_state, ssm_state)
            - conv_state: (batch, d_inner, d_conv)
            - ssm_state: (batch, d_inner, d_state)
        """
        d_inner = self.d_model * self.expand
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        conv_state = torch.zeros(
            batch_size, d_inner, self.d_conv, dtype=dtype, device=device
        )
        ssm_state = torch.zeros(
            batch_size, d_inner, self.d_state, dtype=dtype, device=device
        )

        return conv_state, ssm_state

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, expand={self.expand}"
        )
