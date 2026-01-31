"""
Align-Mamba: SOTA Hybrid Mamba-Attention for Document-Level NMT.

Architecture integrates:
- Polarized Mamba (arXiv:2501.00658): A=0/A=1 channels for recency bias mitigation
- Zamba-style shared GSA (arXiv:2411.15242): Parameter-efficient cross-attention
- Capacity-aware placement (arXiv:2506.11891): Layer 0 fixes "Blind Start"
"""

import math
from typing import Optional, List, Tuple, Dict, Set

import torch
import torch.nn as nn

from .normalization import RMSNorm
from .embeddings import ScaledEmbedding
from .attention import BidirectionalAttention, FlashCrossAttention
from .wrapper import PolarizedMamba2Block
from .bimamba import BiMambaBlock


def compute_cross_attention_positions(
    n_layers: int,
    d_state: int,
    num_pairs: int,
) -> Set[int]:
    """Derive cross-attention placement from capacity theorem.

    Layer 0 always included (Blind Start fix). Additional layers placed
    at intervals based on capacity overflow ratio.

    Reference: arXiv 2506.11891 (Theorem 2), arXiv 2510.03279 (Section 3.1)
    """
    positions = {0}

    if num_pairs <= d_state:
        return positions

    overflow_ratio = num_pairs / d_state
    interval = max(1, int(d_state / math.log(max(overflow_ratio, math.e))))

    for pos in range(interval, n_layers, interval):
        positions.add(pos)

    return positions


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

        # Attention at N/2 and N-1
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

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

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


class HybridMambaDecoder(nn.Module):
    """SOTA Decoder with polarized Mamba and Zamba-style shared cross-attention.

    Architecture:
    - PolarizedMamba2Block: A=0 (local) + A=1 (global) channels fix recency bias
    - Shared GSA: Single cross-attention block with [current, initial] concatenation
    - Capacity-aware placement: Layer 0 + overflow-derived positions
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 128,
        n_heads: int = 12,
        num_pairs: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = ScaledEmbedding(
            vocab_size=vocab_size, d_model=d_model, padding_idx=pad_token_id,
            dropout=dropout, device=device, dtype=dtype,
        )

        # Capacity-aware cross-attention placement
        self.cross_attn_positions = compute_cross_attention_positions(n_layers, d_state, num_pairs)

        factory_kwargs = {"device": device, "dtype": dtype}

        # Polarized Mamba blocks (SOTA: A=0/A=1 channels)
        self.layers = nn.ModuleList([
            PolarizedMamba2Block(
                d_model=d_model, d_state=d_state,
                polarized_channels=2, **factory_kwargs
            )
            for _ in range(n_layers)
        ])

        # Zamba-style shared GSA (SOTA: single shared attention)
        self.shared_cross_attn = FlashCrossAttention(
            d_model=d_model * 2, n_heads=n_heads, dropout=dropout,
            max_seq_len=max_seq_len, **factory_kwargs
        )
        self.cross_attn_projs = nn.ModuleDict({
            str(i): nn.Linear(d_model * 2, d_model, bias=False, **factory_kwargs)
            for i in self.cross_attn_positions
        })

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        x_initial = x  # Zamba: save for GSA concatenation

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

    def init_cache(
        self,
        batch_size: int,
        encoder_out: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict:
        device = device or next(self.parameters()).device
        dtype = dtype or torch.bfloat16

        ssm_states = {}
        for i, layer in enumerate(self.layers):
            ssm_states[i] = layer.allocate_inference_cache(batch_size=batch_size, dtype=dtype, device=device)

        return {"ssm_states": ssm_states, "encoder_output": encoder_out, "seqlen_offset": 0}

    def step(
        self,
        input_ids: torch.Tensor,
        cache: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        x = self.embedding.embed(input_ids) * self.embedding.embed_scale
        if self.embedding.dtype is not None:
            x = x.to(self.embedding.dtype)

        x_initial = x
        offset = cache["seqlen_offset"]

        for i, layer in enumerate(self.layers):
            state = cache["ssm_states"].get(i)
            x = layer(x, inference_params=state) if state else layer(x)

            if i in self.cross_attn_positions and cache["encoder_output"] is not None:
                gsa_input = torch.cat([x, x_initial], dim=-1)
                attn_out = self.shared_cross_attn(gsa_input, cache["encoder_output"], decoder_offset=offset)
                x = x + self.cross_attn_projs[str(i)](attn_out)

        cache["seqlen_offset"] = offset + 1
        return self.lm_head(self.final_norm(x)), cache
