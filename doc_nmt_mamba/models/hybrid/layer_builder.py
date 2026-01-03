"""
Layer builder for hybrid Mamba-Attention architecture.

Implements the 1:7 attention ratio strategy from Jamba:
- Place attention at strategic positions
- Use Mamba/BiMamba for remaining layers

For decoder (from plan):
- Layer 0: HYBRID BLOCK (Mamba + Cross-Attn) - Contextualized Preamble
- Layers 1-7: Mamba (causal)
- Layer 8: HYBRID BLOCK (Mamba + Cross-Attn) - REFRESH 1
- Layers 9-15: Mamba (causal)
- Layer 16: HYBRID BLOCK (Mamba + Cross-Attn) - REFRESH 2
- Layers 17-23: Mamba (causal)

Each HYBRID layer:
    x = x + Mamba(RMSNorm(x))           # Position-aware queries
    x = x + CrossAttn(RMSNorm(x), enc)  # Source-aligned output
"""

from enum import Enum
from typing import List, Tuple, Set, Optional
import warnings

import torch.nn as nn

# Lazy imports for CUDA-dependent modules
_mamba_available = False
Mamba2BlockWrapper = None
BiMambaBlock = None

try:
    from ..mamba2 import Mamba2BlockWrapper as _Mamba2BlockWrapper
    from ..mamba2 import BiMambaBlock as _BiMambaBlock
    Mamba2BlockWrapper = _Mamba2BlockWrapper
    BiMambaBlock = _BiMambaBlock
    _mamba_available = True
except ImportError:
    pass  # Will be handled when building layers

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
    HYBRID = "hybrid"  # Mamba + Cross-Attention in same block


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


class HybridBlock(nn.Module):
    """
    HYBRID Block: Mamba + Cross-Attention.

    From plan - this is CRITICAL for the "Blind Start" fix:
    Layer 0 must be a HYBRID BLOCK (Mamba → Cross-Attention), not just Cross-Attention.
    The Mamba sub-layer creates a "Contextualized Query" so Cross-Attention knows *what* to seek.

    Architecture:
        x = x + Mamba(RMSNorm(x))           # Position-aware, contextualized queries
        x = x + CrossAttn(RMSNorm(x), enc)  # Source-aligned output

    Why Layer 0 HYBRID Block is Essential:
    1. First decoder token sees source immediately
    2. Correct initial alignment → correct state trajectory
    3. Mamba layers 1-7 now have source-informed hidden state
    4. Fits thesis: "Alignment at start + periodic refresh"
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        n_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        layer_idx: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        # Mamba component (comes first to create contextualized queries)
        self.mamba = Mamba2BlockWrapper(
            d_model=d_model,
            d_state=d_state,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

        # Cross-attention component (uses Mamba output as query)
        self.cross_attn = FlashCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x,
        encoder_out,
        decoder_offset: int = 0,
        inference_params=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
    ):
        """
        Forward pass through hybrid block.

        Args:
            x: Decoder hidden states (batch, seq_len, d_model)
            encoder_out: Encoder output (batch, src_len, d_model)
            decoder_offset: Position offset for incremental decoding
            inference_params: Mamba inference state (for generation)
            cu_seqlens_*: For packed sequence mode

        Returns:
            Updated hidden states
        """
        # Step 1: Mamba for position-aware contextualization
        if inference_params is not None:
            x = self.mamba(x, inference_params=inference_params)
        else:
            x = self.mamba(x)

        # Step 2: Cross-attention to encoder
        x = self.cross_attn(
            x,
            encoder_out,
            decoder_offset=decoder_offset,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

        return x


def compute_hybrid_positions(
    n_layers: int,
    hybrid_interval: int = 8,
) -> Set[int]:
    """
    Compute which layer indices should be HYBRID blocks.

    From plan: For 24-layer decoder, hybrid at [0, 8, 16]:
    - Layer 0: Contextualized Preamble (fixes "Blind Start")
    - Layer 8: Refresh 1
    - Layer 16: Refresh 2

    Args:
        n_layers: Total number of layers
        hybrid_interval: Interval between hybrid layers

    Returns:
        Set of layer indices that should be HYBRID blocks
    """
    # Always include layer 0 (critical for "Blind Start" fix)
    positions = {0}

    # Add subsequent layers at regular intervals
    for i in range(hybrid_interval, n_layers, hybrid_interval):
        positions.add(i)

    return positions


def build_decoder_layers(
    n_layers: int,
    d_model: int,
    d_state: int = 128,
    n_heads: int = 8,
    attention_ratio: float = 0.125,
    cross_attn_every: int = 4,
    hybrid_interval: int = 8,
    use_hybrid_blocks: bool = True,
    dropout: float = 0.0,
    max_seq_len: int = 8192,
    device=None,
    dtype=None,
) -> Tuple[nn.ModuleList, List[LayerType]]:
    """
    Build decoder layers with HYBRID blocks at strategic positions.

    NEW ARCHITECTURE (from plan):
    - Layer 0: HYBRID (Mamba + Cross-Attn) - Contextualized Preamble
    - Layers 1-7: Mamba only
    - Layer 8: HYBRID - Refresh 1
    - Layers 9-15: Mamba only
    - Layer 16: HYBRID - Refresh 2
    - Layers 17-23: Mamba only

    Total HYBRID Layers: 3 (at indices [0, 8, 16])
    Ratio: 3/24 = 1:8 = 12.5% (matches Jamba's 1:7 concept)

    Args:
        n_layers: Number of decoder layers
        d_model: Model dimension
        d_state: Mamba state dimension
        n_heads: Number of attention heads
        attention_ratio: Fraction of self-attention layers (for backwards compat)
        cross_attn_every: Deprecated - use hybrid_interval instead
        hybrid_interval: Interval between HYBRID blocks (default 8)
        use_hybrid_blocks: If True, use new HYBRID architecture; else old style
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        device: Device for parameters
        dtype: Data type for parameters

    Returns:
        Tuple of (layers ModuleList, layer_types list)
    """
    if use_hybrid_blocks:
        # NEW: Use HYBRID blocks at [0, 8, 16, ...]
        hybrid_positions = compute_hybrid_positions(n_layers, hybrid_interval)

        layers = nn.ModuleList()
        layer_types = []

        for i in range(n_layers):
            if i in hybrid_positions:
                # HYBRID: Mamba + Cross-Attention in same block
                layer = HybridBlock(
                    d_model=d_model,
                    d_state=d_state,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    layer_idx=i,
                    device=device,
                    dtype=dtype,
                )
                layers.append(layer)
                layer_types.append(LayerType.HYBRID)
            else:
                # Pure Mamba layer
                layer = Mamba2BlockWrapper(
                    d_model=d_model,
                    d_state=d_state,
                    layer_idx=i,
                    device=device,
                    dtype=dtype,
                )
                layers.append(layer)
                layer_types.append(LayerType.MAMBA)

        return layers, layer_types

    # OLD STYLE: For backwards compatibility
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
