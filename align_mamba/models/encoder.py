"""HybridBiMambaEncoder: Bidirectional Mamba with sparse attention.

Architecture:
- BiMambaBlock: Forward + backward Mamba scans for bidirectional context
- BidirectionalAttention at layers N/2 and N-1 for global context
"""

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from mamba_ssm import Mamba2

from .components import RMSNorm, ScaledEmbedding
from .attention import BidirectionalAttention
from .segment_utils import process_segments


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba: concatenate OUTPUTS (y), not internal states (h).

    Forward + backward scans on full d_model, concat to 2*d_model, project down.
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

        if cu_seqlens is not None:
            y_fwd, y_bwd = process_segments(x, self.mamba_fwd, cu_seqlens, self.mamba_bwd)
        else:
            y_fwd = self.mamba_fwd(x)
            x_flipped = torch.flip(x, dims=[1])
            y_bwd = torch.flip(self.mamba_bwd(x_flipped), dims=[1])

        out = self.out_proj(torch.cat([y_fwd, y_bwd], dim=-1))
        return residual + out


class HybridBiMambaEncoder(nn.Module):
    """BiMamba encoder with sparse attention at N/2 and N-1 layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 16,
        d_state: int = 128,
        n_heads: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = ScaledEmbedding(
            vocab_size=vocab_size, d_model=d_model, padding_idx=pad_token_id,
            dropout=dropout, device=device, dtype=dtype,
        )

        self.attention_positions = {n_layers // 2, n_layers - 1}

        factory_kwargs = {"device": device, "dtype": dtype}
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in self.attention_positions:
                self.layers.append(BidirectionalAttention(
                    d_model=d_model, n_heads=n_heads, dropout=dropout,
                    max_seq_len=max_seq_len, **factory_kwargs,
                ))
            else:
                self.layers.append(BiMambaBlock(
                    d_model=d_model, d_state=d_state, **factory_kwargs,
                ))

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)

        for i, layer in enumerate(self.layers):
            is_attention = i in self.attention_positions
            if self._gradient_checkpointing and self.training:
                if is_attention:
                    x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
                else:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x, attention_mask=attention_mask) if is_attention else layer(x)

        return self.final_norm(x)
