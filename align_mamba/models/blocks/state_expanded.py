"""HGRN2 State Expansion: outer product d -> d^2 capacity (arXiv:2404.07904)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from align_mamba.models.components import RMSNorm
from align_mamba.models.registry import BlockRegistry


@BlockRegistry.register("state_expanded")
class StateExpandedBlock(nn.Module):
    """HGRN2-style d -> d^2 state expansion (arXiv:2404.07904)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        head_dim: int = 128,
        forget_lower_bound: float = 0.9,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,  # Accept extra kwargs for registry compatibility
    ):
        super().__init__()

        # Allow flexible head configuration
        if d_model % head_dim == 0:
            n_heads = d_model // head_dim
        else:
            # Fall back to n_heads if head_dim doesn't divide evenly
            head_dim = d_model // n_heads

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.forget_lower_bound = forget_lower_bound

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Projections for gates and values
        self.forget_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.input_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.output_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # Initialize forget gate bias toward 1 (preserving information)
        nn.init.zeros_(self.forget_proj.weight)

    def _forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized outer-product recurrence via cumsum/cumprod."""
        B, T, D = x.shape

        f_raw = self.forget_proj(x)
        f_t = torch.clamp(torch.sigmoid(f_raw), min=self.forget_lower_bound)
        i_t = 1 - f_t

        v_t = self.input_proj(x)
        o_t = torch.sigmoid(self.output_proj(x))

        f_t = f_t.view(B, T, self.n_heads, self.head_dim)
        i_t = i_t.view(B, T, self.n_heads, self.head_dim)
        v_t = v_t.view(B, T, self.n_heads, self.head_dim)
        o_t = o_t.view(B, T, self.n_heads, self.head_dim)

        outer = i_t.unsqueeze(-1) * v_t.unsqueeze(-2)

        # Log-space cumulative products for numerical stability
        log_f = torch.log(f_t + 1e-8)
        log_cumprod_f = torch.cumsum(log_f, dim=1)

        log_F_expanded = log_cumprod_f.unsqueeze(-1)
        outer_weighted = outer * torch.exp(-log_F_expanded)
        S = torch.cumsum(outer_weighted, dim=1)
        h = S * torch.exp(log_F_expanded)

        output = (o_t.unsqueeze(-1) * h).sum(dim=-1)
        return output.view(B, T, D)

    def forward(
        self,
        x: torch.Tensor,
        inference_params: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, d_model)
            inference_params: Optional state for autoregressive decoding (B, n_heads, head_dim, head_dim)
            cu_seqlens: Optional cumulative sequence lengths for packed sequences

        Returns:
            Output tensor (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = x.contiguous()

        # Use reference implementation (CUDA kernel can be added later)
        out = self._forward_reference(x)

        return residual + out

    def allocate_inference_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate state cache for autoregressive decoding.

        Returns:
            Tuple of (conv_state_placeholder, ssm_state) for interface compatibility
            Note: conv_state is not used, but returned for API compatibility with PolarizedMamba2Block
        """
        dtype = dtype or torch.bfloat16
        device = device or next(self.parameters()).device

        # Placeholder for conv_state (not used in state expansion)
        conv_state = torch.zeros(1, device=device, dtype=dtype)

        # Actual state: (B, n_heads, head_dim, head_dim)
        ssm_state = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.head_dim,
            dtype=dtype, device=device
        )

        return conv_state, ssm_state


def compute_forget_lower_bound(layer_idx: int, n_layers: int) -> float:
    """Depth-dependent forget gate bound (0.9 to 0.99)."""
    return 0.9 + 0.09 * (layer_idx / max(n_layers - 1, 1))
