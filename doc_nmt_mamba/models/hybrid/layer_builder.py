"""
Layer builder for hybrid Mamba-Attention architecture.

Implements the 1:7 attention ratio strategy from Jamba:
- Place attention at strategic positions (middle + final)
- Use Mamba/BiMamba for remaining layers
- Add cross-attention every N decoder layers
"""

from enum import Enum
from typing import List, Tuple, Set, Optional

import torch.nn as nn

from ..mamba2 import Mamba2BlockWrapper, BiMambaBlock
from ..attention import (
    BidirectionalAttention,
    CausalSelfAttention,
    FlashCrossAttention,
)


class LayerType(Enum):
    """Types of layers in the hybrid architecture."""

    MAMBA = "mamba"
    BIMAMBA = "bimamba"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"


def compute_attention_positions(
    n_layers: int,
    attention_ratio: float = 0.125,  # 1:7 ratio
) -> Set[int]:
    """
    Compute which layer indices should have attention.

    Strategy (based on Jamba/Mamba-MHA research):
    - Middle layer (N/2): captures bidirectional context
    - Final layer (N-1): output refinement
    - Additional positions distributed evenly if ratio requires more

    Special case: attention_ratio >= 1.0 means ALL layers are attention
    (pure Transformer baseline for comparison experiments)

    Args:
        n_layers: Total number of layers
        attention_ratio: Fraction of layers that are attention

    Returns:
        Set of layer indices that should be attention layers
    """
    # Special case: pure Transformer (all attention)
    if attention_ratio >= 1.0:
        return set(range(n_layers))

    n_attention = max(2, int(n_layers * attention_ratio))

    # Always include middle and final positions
    positions = {n_layers // 2, n_layers - 1}

    # If we need more attention layers, distribute evenly
    if n_attention > 2:
        remaining = n_attention - 2
        step = n_layers // (remaining + 1)
        for i in range(1, remaining + 1):
            pos = i * step
            if pos not in positions:
                positions.add(pos)

    # Ensure we don't exceed the requested number
    while len(positions) > n_attention:
        # Remove positions that aren't middle or final
        for p in list(positions):
            if p not in {n_layers // 2, n_layers - 1}:
                positions.remove(p)
                break

    return positions


def build_encoder_layers(
    n_layers: int,
    d_model: int,
    d_state: int = 128,
    n_heads: int = 8,
    attention_ratio: float = 0.125,
    dropout: float = 0.0,
    max_seq_len: int = 8192,
    device=None,
    dtype=None,
) -> Tuple[nn.ModuleList, List[LayerType]]:
    """
    Build encoder layers with BiMamba + sparse bidirectional attention.

    Args:
        n_layers: Number of encoder layers
        d_model: Model dimension
        d_state: Mamba state dimension
        n_heads: Number of attention heads
        attention_ratio: Fraction of attention layers (1/8 = 1:7 ratio)
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        device: Device for parameters
        dtype: Data type for parameters

    Returns:
        Tuple of (layers ModuleList, layer_types list)
    """
    attention_positions = compute_attention_positions(n_layers, attention_ratio)

    layers = nn.ModuleList()
    layer_types = []

    for i in range(n_layers):
        if i in attention_positions:
            # Bidirectional attention for encoder
            layer = BidirectionalAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype,
            )
            layers.append(layer)
            layer_types.append(LayerType.ATTENTION)
        else:
            # BiMamba for bidirectional context
            layer = BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                device=device,
                dtype=dtype,
            )
            layers.append(layer)
            layer_types.append(LayerType.BIMAMBA)

    return layers, layer_types


def build_decoder_layers(
    n_layers: int,
    d_model: int,
    d_state: int = 128,
    n_heads: int = 8,
    attention_ratio: float = 0.125,
    cross_attn_every: int = 4,
    dropout: float = 0.0,
    max_seq_len: int = 8192,
    device=None,
    dtype=None,
) -> Tuple[nn.ModuleList, List[LayerType]]:
    """
    Build decoder layers with causal Mamba + sparse self-attention + cross-attention.

    Args:
        n_layers: Number of decoder layers
        d_model: Model dimension
        d_state: Mamba state dimension
        n_heads: Number of attention heads
        attention_ratio: Fraction of attention layers
        cross_attn_every: Add cross-attention every N layers
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        device: Device for parameters
        dtype: Data type for parameters

    Returns:
        Tuple of (layers ModuleList, layer_types list)
    """
    attention_positions = compute_attention_positions(n_layers, attention_ratio)

    layers = nn.ModuleList()
    layer_types = []

    for i in range(n_layers):
        # Self-attention or Mamba layer
        if i in attention_positions:
            # Causal self-attention
            layer = CausalSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype,
            )
            layers.append(layer)
            layer_types.append(LayerType.ATTENTION)
        else:
            # Causal Mamba
            layer = Mamba2BlockWrapper(
                d_model=d_model,
                d_state=d_state,
                layer_idx=i,
                device=device,
                dtype=dtype,
            )
            layers.append(layer)
            layer_types.append(LayerType.MAMBA)

        # Add cross-attention every N layers
        if (i + 1) % cross_attn_every == 0:
            cross_attn = FlashCrossAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype,
            )
            layers.append(cross_attn)
            layer_types.append(LayerType.CROSS_ATTENTION)

    return layers, layer_types


def count_layer_types(layer_types: List[LayerType]) -> dict:
    """Count the number of each layer type."""
    counts = {}
    for lt in LayerType:
        counts[lt.value] = sum(1 for t in layer_types if t == lt)
    return counts
