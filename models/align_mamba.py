"""
Align-Mamba: Hybrid Mamba-Attention blocks for Document-Level NMT.

HYBRID blocks at decoder [0, 8, 16] fix "Blind Start": Mamba creates contextualized
query before cross-attention, ensuring correct initial alignment.
"""

import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union, Set

import torch
import torch.nn as nn

from .normalization import RMSNorm
from .attention import BidirectionalAttention, FlashCrossAttention
from .wrapper import Mamba2BlockWrapper
from .bimamba import BiMambaBlock


class LayerType(Enum):
    BIMAMBA = "bimamba"
    ATTENTION = "attention"


def count_layer_types(layer_types: List[LayerType]) -> dict:
    counts = {}
    for lt in LayerType:
        counts[lt.value] = sum(1 for t in layer_types if t == lt)
    return counts


def compute_hybrid_positions(n_layers: int, num_hybrid: int = 3) -> Set[int]:
    """
    Compute scale-invariant hybrid positions for decoder.

    Formula: [0, N//3, 2N//3] for N layers.
    - Layer 0 always included (fixes Blind Start problem)
    - Remaining positions evenly distributed for periodic source refresh

    Args:
        n_layers: Total number of decoder layers
        num_hybrid: Target number of hybrid blocks (default 3)

    Returns:
        Set of layer indices to be hybrid blocks
    """
    if n_layers <= num_hybrid:
        return set(range(n_layers))

    positions = {0}  # Layer 0 always included
    step = n_layers // num_hybrid
    for i in range(1, num_hybrid):
        positions.add(i * step)
    return positions


@dataclass
class MambaInferenceState:
    conv_state: torch.Tensor  # (batch, d_inner, d_conv)
    ssm_state: torch.Tensor   # (batch, d_inner, d_state)


class HybridBiMambaEncoder(nn.Module):
    """
    BiMamba encoder with sparse attention at middle (N/2) and final (N-1) layers.
    BiMamba provides O(L) bidirectional context; attention enables in-context learning.

    Attention ratio is adaptive when None: guarantees exactly 2 attention layers (2/n_layers).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 16,
        d_state: int = 128,
        n_heads: int = 12,
        attention_ratio: Optional[float] = None,  # None = adaptive (2/n_layers)
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_token_id = pad_token_id
        self.dtype = dtype

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, device=device)
        # Learned embedding scale (initialized to sqrt(d_model))
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(d_model)))
        self.embed_dropout = nn.Dropout(dropout)

        # Adaptive attention ratio: 2/n_layers ensures exactly 2 attention layers
        if attention_ratio is None:
            attention_ratio = 2.0 / n_layers
        self.attention_positions = self._compute_attention_positions(n_layers, attention_ratio)

        self.layers = nn.ModuleList()
        self.layer_types = []

        factory_kwargs = {"device": device, "dtype": dtype}

        for i in range(n_layers):
            if i in self.attention_positions:
                layer = BidirectionalAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    **factory_kwargs,
                )
                self.layers.append(layer)
                self.layer_types.append(LayerType.ATTENTION)
            else:
                layer = BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    **factory_kwargs,
                )
                self.layers.append(layer)
                self.layer_types.append(LayerType.BIMAMBA)

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self._gradient_checkpointing = False

    def _compute_attention_positions(self, n_layers: int, attention_ratio: float) -> Set[int]:
        """Place attention at middle (N/2) and final (N-1) layers."""
        if attention_ratio >= 1.0:
            return set(range(n_layers))

        n_attention = max(2, int(n_layers * attention_ratio))
        positions = {n_layers // 2, n_layers - 1}

        if n_attention > 2:
            remaining = n_attention - 2
            step = n_layers // (remaining + 1)
            for i in range(1, remaining + 1):
                pos = i * step
                if pos not in positions:
                    positions.add(pos)

        while len(positions) > n_attention:
            for p in list(positions):
                if p not in {n_layers // 2, n_layers - 1}:
                    positions.remove(p)
                    break

        return positions

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        x = self.embed_dropout(x)

        for layer, layer_type in zip(self.layers, self.layer_types):
            if self._gradient_checkpointing and self.training:
                if layer_type == LayerType.ATTENTION:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, attention_mask, use_reentrant=False
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
            else:
                if layer_type == LayerType.ATTENTION:
                    x = layer(x, attention_mask=attention_mask)
                else:
                    x = layer(x)

        x = self.final_norm(x)
        return x

    def get_layer_counts(self) -> dict:
        return count_layer_types(self.layer_types)

    def extra_repr(self) -> str:
        counts = self.get_layer_counts()
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"bimamba={counts['bimamba']}, attention={counts['attention']}"
        )


class HybridMambaDecoder(nn.Module):
    """
    Decoder with cross-attention at adaptive positions. Layer 0 cross-attention fixes "Blind Start",
    additional cross-attention layers provide periodic source refresh. All layers use Mamba for O(1) cached decoding.

    Default positions computed via scale-invariant formula: [0, N//3, 2N//3] for N layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 128,
        n_heads: int = 12,
        hybrid_positions: Optional[List[int]] = None,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_token_id = pad_token_id
        self.dtype = dtype

        self.expand = 2
        self.d_conv = 4

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, device=device)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(d_model)))
        self.embed_dropout = nn.Dropout(dropout)

        # Scale-invariant hybrid positions: [0, N//3, 2N//3] for N layers
        if hybrid_positions is not None:
            self.hybrid_positions = set(hybrid_positions)
        else:
            self.hybrid_positions = compute_hybrid_positions(n_layers)

        factory_kwargs = {"device": device, "dtype": dtype}

        # All layers are Mamba blocks
        self.layers = nn.ModuleList([
            Mamba2BlockWrapper(d_model=d_model, d_state=d_state, layer_idx=i, **factory_kwargs)
            for i in range(n_layers)
        ])

        # Cross-attention only at hybrid positions
        self.cross_attn = nn.ModuleDict({
            str(i): FlashCrossAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
                max_seq_len=max_seq_len, **factory_kwargs
            )
            for i in self.hybrid_positions
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
        attention_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        x = self.embed_dropout(x)

        for i, layer in enumerate(self.layers):
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

            # Cross-attention at hybrid positions (Mamba first creates contextualized query)
            if str(i) in self.cross_attn and encoder_out is not None:
                if self._gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        self.cross_attn[str(i)], x, encoder_out, encoder_padding_mask,
                        use_reentrant=False
                    )
                else:
                    x = self.cross_attn[str(i)](x, encoder_out, encoder_padding_mask=encoder_padding_mask)

        x = self.final_norm(x)
        return self.lm_head(x)

    def init_cache(
        self,
        batch_size: int,
        encoder_out: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Union[Dict[int, MambaInferenceState], torch.Tensor, int]]:
        device = device or next(self.parameters()).device
        dtype = dtype or torch.bfloat16

        ssm_states = {}
        for i, layer in enumerate(self.layers):
            conv_state, ssm_state = layer.allocate_inference_cache(
                batch_size=batch_size, dtype=dtype, device=device
            )
            ssm_states[i] = MambaInferenceState(conv_state, ssm_state)

        return {"ssm_states": ssm_states, "encoder_output": encoder_out, "seqlen_offset": 0}

    def step(
        self,
        input_ids: torch.Tensor,
        cache: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)

        offset = cache["seqlen_offset"]

        for i, layer in enumerate(self.layers):
            state = cache["ssm_states"].get(i)
            if state is not None:
                x = layer(x, inference_params=(state.conv_state, state.ssm_state))
            else:
                x = layer(x)

            if str(i) in self.cross_attn and cache["encoder_output"] is not None:
                x = self.cross_attn[str(i)](
                    x, cache["encoder_output"], decoder_offset=offset
                )

        cache["seqlen_offset"] = offset + 1
        x = self.final_norm(x)
        return self.lm_head(x), cache

    def get_layer_counts(self) -> dict:
        return {"mamba": self.n_layers, "cross_attn": len(self.cross_attn)}

    def extra_repr(self) -> str:
        counts = self.get_layer_counts()
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"mamba={counts['mamba']}, cross_attn={counts['cross_attn']}"
        )
