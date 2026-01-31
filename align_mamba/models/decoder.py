"""HybridMambaDecoder with Polarized Mamba and Zamba-style shared cross-attention."""

import math
from typing import Optional, Set, List, TYPE_CHECKING

import torch
import torch.nn as nn

from .components import RMSNorm, ScaledEmbedding
from .registry import BlockRegistry, AttentionRegistry
from . import blocks  # noqa: F401 - triggers registration
from .blocks.state_expanded import compute_forget_lower_bound

if TYPE_CHECKING:
    from align_mamba.config import SOTAConfig


def compute_cross_attention_positions(n_layers: int, d_state: int, num_pairs: int) -> Set[int]:
    """Layer 0 + overflow-derived positions (arXiv 2506.11891)."""
    positions = {0}

    if num_pairs <= d_state:
        return positions

    overflow_ratio = num_pairs / d_state
    interval = max(1, int(d_state / math.log(max(overflow_ratio, math.e))))

    for pos in range(interval, n_layers, interval):
        positions.add(pos)

    return positions


class HybridMambaDecoder(nn.Module):
    """Decoder with polarized Mamba and Zamba-style shared cross-attention."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 128,
        n_heads: int = 12,
        num_pairs: int = 64,
        hybrid_positions: Optional[List[int]] = None,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        sota_config: Optional["SOTAConfig"] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_state = d_state

        if sota_config is None:
            from align_mamba.config import SOTAConfig
            sota_config = SOTAConfig()

        self.embedding = ScaledEmbedding(
            vocab_size=vocab_size, d_model=d_model, padding_idx=pad_token_id,
            dropout=dropout, device=device, dtype=dtype,
        )

        if hybrid_positions is not None:
            self.cross_attn_positions = set(hybrid_positions)
        else:
            self.cross_attn_positions = compute_cross_attention_positions(n_layers, d_state, num_pairs)

        factory_kwargs = {"device": device, "dtype": dtype}

        block_type = sota_config.block_type.value if hasattr(sota_config.block_type, 'value') else sota_config.block_type
        block_cls = BlockRegistry.get(block_type)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            block_kwargs = {
                "d_model": d_model,
                "d_state": d_state,
                "head_dim": sota_config.state_expansion_head_dim,
                "forget_lower_bound": compute_forget_lower_bound(i, n_layers),
                "pool_size": sota_config.memmamba_pool_size,
                "summary_dim": sota_config.memmamba_summary_dim,
                "tau1": sota_config.memmamba_tau1,
                "tau2": sota_config.memmamba_tau2,
                "layer_idx": i,
                "cross_layer_frequency": sota_config.memmamba_cross_layer_freq,
                **factory_kwargs,
            }
            self.layers.append(block_cls(**block_kwargs))

        attention_type = sota_config.attention_type.value if hasattr(sota_config.attention_type, 'value') else sota_config.attention_type
        attn_cls = AttentionRegistry.get(attention_type)

        attn_kwargs = {
            "d_model": d_model * 2,
            "n_heads": n_heads,
            "dropout": dropout,
            "max_seq_len": max_seq_len,
            "feature_dim": sota_config.based_feature_dim,
            "window_size": sota_config.based_window_size,
            **factory_kwargs,
        }
        self.shared_cross_attn = attn_cls(**attn_kwargs)

        self.cross_attn_projs = nn.ModuleDict({
            str(i): nn.Linear(d_model * 2, d_model, bias=False, **factory_kwargs)
            for i in self.cross_attn_positions
        })

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through decoder."""
        x = self.embedding(input_ids)
        x_initial = x

        for i, layer in enumerate(self.layers):
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

            if i in self.cross_attn_positions and encoder_out is not None:
                gsa_input = torch.cat([x, x_initial], dim=-1)
                if self._gradient_checkpointing and self.training:
                    attn_out = torch.utils.checkpoint.checkpoint(
                        self.shared_cross_attn, gsa_input, encoder_out, encoder_padding_mask, use_reentrant=False
                    )
                else:
                    attn_out = self.shared_cross_attn(gsa_input, encoder_out, encoder_padding_mask=encoder_padding_mask)

                x = x + self.cross_attn_projs[str(i)](attn_out)

        return self.lm_head(self.final_norm(x))
