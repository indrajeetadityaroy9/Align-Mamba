"""
Hybrid Mamba-Attention Architecture for Document-Level NMT.

This file contains:
- LayerType enum and layer counting utilities
- HybridBlock: Combined Mamba + Cross-Attention block
- HybridBiMambaEncoder: Bidirectional encoder
- HybridMambaDecoder: Causal decoder with HYBRID blocks at [0, 8, 16]
- HybridMambaEncoderDecoder: Full encoder-decoder model
- MambaState, AttentionKVCache: Inference state management
- ModelConfig: Model configuration dataclass

CRITICAL ARCHITECTURE DECISIONS:
1. HYBRID blocks at decoder layers [0, 8, 16] (explicit, not computed)
2. Each HYBRID block: Mamba first (creates contextualized query), then Cross-Attention
3. Layer 0 HYBRID fixes "Blind Start" problem
4. BiMamba encoder with sparse bidirectional attention at strategic positions
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Set

import torch
import torch.nn as nn

from .layers import (
    RMSNorm,
    BidirectionalAttention,
    CausalSelfAttention,
    FlashCrossAttention,
    Mamba2BlockWrapper,
    BiMambaBlock,
)


# =============================================================================
# Layer Type Enum
# =============================================================================

class LayerType(Enum):
    """Types of layers in the hybrid architecture."""
    MAMBA = "mamba"
    BIMAMBA = "bimamba"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"
    HYBRID = "hybrid"  # Mamba + Cross-Attention in same block


def count_layer_types(layer_types: List[LayerType]) -> dict:
    """Count the number of each layer type."""
    counts = {}
    for lt in LayerType:
        counts[lt.value] = sum(1 for t in layer_types if t == lt)
    return counts


# =============================================================================
# Inference State Dataclasses
# =============================================================================

@dataclass
class MambaState:
    """State for a Mamba layer during inference."""
    conv_state: torch.Tensor  # (batch, d_inner, d_conv)
    ssm_state: torch.Tensor   # (batch, d_inner, d_state)


@dataclass
class AttentionKVCache:
    """KV cache for an attention layer during inference."""
    key_cache: Optional[torch.Tensor]    # (batch, seq_len, n_heads, head_dim)
    value_cache: Optional[torch.Tensor]  # (batch, seq_len, n_heads, head_dim)


@dataclass
class HybridCacheParams:
    """
    Hybrid cache for Mamba + Attention layers during autoregressive generation.

    CRITICAL: This structure must be correct from Day 1!
    - Mamba: Fixed-size state (B, d_model*expand, d_state)
    - Attention: Growing KV cache (B, current_len, n_heads, head_dim)
    """
    ssm_states: Dict[int, MambaState] = field(default_factory=dict)
    kv_caches: Dict[int, AttentionKVCache] = field(default_factory=dict)
    encoder_output: Optional[torch.Tensor] = None
    seqlen_offset: int = 0


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the Hybrid Mamba-Attention model.

    ARCHITECTURE:
    - Decoder uses HYBRID blocks at [0, 8, 16] for 24-layer model
    - Each HYBRID block contains Mamba + Cross-Attention
    - Ratio: 3/24 = 1:8 = 12.5%
    """
    vocab_size: int = 32000
    d_model: int = 768
    encoder_layers: int = 16
    decoder_layers: int = 24  # For [0, 8, 16] pattern
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 12
    attention_ratio: float = 0.125  # 1:7 ratio for encoder
    hybrid_interval: int = 8  # Interval between HYBRID blocks
    dropout: float = 0.1
    max_seq_len: int = 8192
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# =============================================================================
# Hybrid Block (Mamba + Cross-Attention)
# =============================================================================

class HybridBlock(nn.Module):
    """
    HYBRID Block: Mamba + Cross-Attention.

    From plan - this is CRITICAL for the "Blind Start" fix:
    Layer 0 must be a HYBRID BLOCK (Mamba -> Cross-Attention), not just Cross-Attention.
    The Mamba sub-layer creates a "Contextualized Query" so Cross-Attention knows *what* to seek.

    Architecture:
        x = x + Mamba(RMSNorm(x))           # Position-aware, contextualized queries
        x = x + CrossAttn(RMSNorm(x), enc)  # Source-aligned output

    Why Layer 0 HYBRID Block is Essential:
    1. First decoder token sees source immediately
    2. Correct initial alignment -> correct state trajectory
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


# =============================================================================
# Hybrid BiMamba Encoder
# =============================================================================

class HybridBiMambaEncoder(nn.Module):
    """
    Hybrid encoder with BiMamba + sparse bidirectional attention.

    BiMamba provides bidirectional context with O(L) complexity.
    Strategic attention layers (1:7 ratio) enable in-context learning.

    Attention layers are placed at:
    - Middle layer (N/2): captures bidirectional context
    - Final layer (N-1): output refinement
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 16,
        d_state: int = 128,
        n_heads: int = 12,
        attention_ratio: float = 0.125,
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

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, device=device)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Compute attention positions explicitly
        # For encoder: place at middle and final positions
        self.attention_positions = self._compute_attention_positions(n_layers, attention_ratio)

        # Build layers explicitly (no factory function)
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

        # Final normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self._gradient_checkpointing = False

    def _compute_attention_positions(self, n_layers: int, attention_ratio: float) -> Set[int]:
        """Compute which layer indices should have attention.

        Strategy: middle layer (N/2) and final layer (N-1).
        """
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
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input sequence.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask (batch, seq_len)

        Returns:
            Encoder output (batch, seq_len, d_model)
        """
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
        """Get count of each layer type."""
        return count_layer_types(self.layer_types)

    def extra_repr(self) -> str:
        counts = self.get_layer_counts()
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"bimamba={counts['bimamba']}, attention={counts['attention']}"
        )


# =============================================================================
# Hybrid Mamba Decoder
# =============================================================================

class HybridMambaDecoder(nn.Module):
    """
    Hybrid decoder with HYBRID blocks at strategic positions.

    ARCHITECTURE (explicit, not computed):
    - Layer 0: HYBRID (Mamba + Cross-Attn) - Contextualized Preamble
    - Layers 1-7: Mamba only
    - Layer 8: HYBRID - Refresh 1
    - Layers 9-15: Mamba only
    - Layer 16: HYBRID - Refresh 2
    - Layers 17-23: Mamba only

    Total HYBRID Layers: 3 (at indices [0, 8, 16])
    Ratio: 3/24 = 1:8 = 12.5%

    Complexity per generation step:
    - Mamba: O(1) via state caching
    - Cross-attention: O(L_src) at HYBRID layers only
    """

    # EXPLICIT hybrid positions - not computed!
    DEFAULT_HYBRID_POSITIONS = {0, 8, 16}

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 128,
        n_heads: int = 12,
        hybrid_interval: int = 8,
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
        self.hybrid_interval = hybrid_interval
        self.dtype = dtype

        # For state management during inference
        self.expand = 2
        self.d_conv = 4

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, device=device)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # EXPLICIT hybrid positions: [0, 8, 16, ...]
        self.hybrid_positions = {0}
        for i in range(hybrid_interval, n_layers, hybrid_interval):
            self.hybrid_positions.add(i)

        # Build layers EXPLICITLY
        self.layers = nn.ModuleList()
        self.layer_types = []

        factory_kwargs = {"device": device, "dtype": dtype}

        for i in range(n_layers):
            if i in self.hybrid_positions:
                # HYBRID: Mamba + Cross-Attention in same block
                layer = HybridBlock(
                    d_model=d_model,
                    d_state=d_state,
                    n_heads=n_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    layer_idx=i,
                    **factory_kwargs,
                )
                self.layers.append(layer)
                self.layer_types.append(LayerType.HYBRID)
            else:
                # Pure Mamba layer
                layer = Mamba2BlockWrapper(
                    d_model=d_model,
                    d_state=d_state,
                    layer_idx=i,
                    **factory_kwargs,
                )
                self.layers.append(layer)
                self.layer_types.append(LayerType.MAMBA)

        # Final normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Output projection (language model head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decoder forward pass (training mode).

        Args:
            input_ids: Target token IDs (batch, seq_len)
            encoder_out: Encoder output (batch, src_len, d_model)
            attention_mask: Optional attention mask

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        x = self.embed_dropout(x)

        for layer, layer_type in zip(self.layers, self.layer_types):
            if self._gradient_checkpointing and self.training:
                if layer_type == LayerType.HYBRID:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, encoder_out, use_reentrant=False
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
            else:
                if layer_type == LayerType.HYBRID:
                    x = layer(x, encoder_out)
                else:
                    x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    def init_cache(
        self,
        batch_size: int,
        encoder_out: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Union[Dict[int, MambaState], Dict[int, AttentionKVCache], torch.Tensor, int]]:
        """
        Initialize inference cache for autoregressive generation.

        Args:
            batch_size: Batch size
            encoder_out: Encoder output (cached for cross-attention)
            device: Device for cache tensors
            dtype: Data type for cache tensors

        Returns:
            Cache dictionary
        """
        device = device or next(self.parameters()).device
        dtype = dtype or torch.bfloat16

        ssm_states = {}
        kv_caches = {}

        for layer_idx, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == LayerType.MAMBA:
                conv_state, ssm_state = layer.allocate_inference_cache(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
                ssm_states[layer_idx] = MambaState(conv_state, ssm_state)
            elif layer_type == LayerType.HYBRID:
                conv_state, ssm_state = layer.mamba.allocate_inference_cache(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                )
                ssm_states[layer_idx] = MambaState(conv_state, ssm_state)

        return {
            "ssm_states": ssm_states,
            "kv_caches": kv_caches,
            "encoder_output": encoder_out,
            "seqlen_offset": 0,
        }

    def step(
        self,
        input_ids: torch.Tensor,
        cache: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single token generation step.

        Args:
            input_ids: Current token IDs (batch, 1)
            cache: Inference cache from init_cache or previous step

        Returns:
            Tuple of (logits, updated_cache)
        """
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)

        offset = cache["seqlen_offset"]

        for layer_idx, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == LayerType.MAMBA:
                state = cache["ssm_states"].get(layer_idx)
                if state is not None:
                    x = layer(x, inference_params=(state.conv_state, state.ssm_state))
                else:
                    x = layer(x)
            elif layer_type == LayerType.HYBRID:
                state = cache["ssm_states"].get(layer_idx)
                inference_params = (state.conv_state, state.ssm_state) if state else None
                x = layer(
                    x,
                    cache["encoder_output"],
                    decoder_offset=offset,
                    inference_params=inference_params,
                )

        cache["seqlen_offset"] = offset + 1

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, cache

    def get_layer_counts(self) -> dict:
        """Get count of each layer type."""
        return count_layer_types(self.layer_types)

    def extra_repr(self) -> str:
        counts = self.get_layer_counts()
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"mamba={counts['mamba']}, hybrid={counts['hybrid']}"
        )


# =============================================================================
# Full Encoder-Decoder Model
# =============================================================================

class HybridMambaEncoderDecoder(nn.Module):
    """
    Full Encoder-Decoder model with Hybrid Mamba-Attention architecture.

    Features:
    - BiMamba encoder with sparse bidirectional attention
    - Causal Mamba decoder with HYBRID blocks at [0, 8, 16]
    - Efficient autoregressive generation with hybrid state management
    - Gradient checkpointing support for long sequences

    Target performance (200M params on H100):
    - Training: batch_size=64, seq_len=8192
    - Inference: O(L_src) + O(1) per token (vs O(L_src + L_tgt) for Transformer)
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        vocab_size: int = 32000,
        d_model: int = 768,
        encoder_layers: int = 16,
        decoder_layers: int = 24,
        d_state: int = 128,
        n_heads: int = 12,
        attention_ratio: float = 0.125,
        hybrid_interval: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        share_embeddings: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Use config if provided
        if config is not None:
            vocab_size = config.vocab_size
            d_model = config.d_model
            encoder_layers = config.encoder_layers
            decoder_layers = config.decoder_layers
            d_state = config.d_state
            n_heads = config.n_heads
            attention_ratio = config.attention_ratio
            hybrid_interval = config.hybrid_interval
            dropout = config.dropout
            max_seq_len = config.max_seq_len
            pad_token_id = config.pad_token_id
            bos_token_id = config.bos_token_id
            eos_token_id = config.eos_token_id

        # Store config
        self.config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            d_state=d_state,
            n_heads=n_heads,
            attention_ratio=attention_ratio,
            hybrid_interval=hybrid_interval,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        factory_kwargs = {"device": device, "dtype": dtype}

        # Encoder
        self.encoder = HybridBiMambaEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=encoder_layers,
            d_state=d_state,
            n_heads=n_heads,
            attention_ratio=attention_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
            **factory_kwargs,
        )

        # Decoder with HYBRID blocks
        self.decoder = HybridMambaDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=decoder_layers,
            d_state=d_state,
            n_heads=n_heads,
            hybrid_interval=hybrid_interval,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
            **factory_kwargs,
        )

        # Optionally share embeddings
        if share_embeddings:
            self.decoder.embed.weight = self.encoder.embed.weight

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src_ids: Source token IDs (batch, src_len)
            tgt_ids: Target token IDs (batch, tgt_len)
            src_mask: Optional source attention mask
            tgt_mask: Optional target attention mask

        Returns:
            Logits (batch, tgt_len, vocab_size)
        """
        encoder_out = self.encoder(src_ids, attention_mask=src_mask)
        logits = self.decoder(tgt_ids, encoder_out, attention_mask=tgt_mask)
        return logits

    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source sequence."""
        return self.encoder(src_ids, attention_mask=src_mask)

    def init_generation_cache(
        self,
        encoder_out: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict:
        """Initialize cache for autoregressive generation."""
        batch_size = encoder_out.size(0)
        return self.decoder.init_cache(batch_size, encoder_out, device, dtype)

    def generate_step(
        self,
        input_ids: torch.Tensor,
        cache: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """Single generation step."""
        return self.decoder.step(input_ids, cache)

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with greedy/sampling decoding.

        Args:
            src_ids: Source token IDs (batch, src_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            src_mask: Optional source attention mask

        Returns:
            Generated token IDs (batch, gen_len)
        """
        batch_size = src_ids.size(0)
        device = src_ids.device

        encoder_out = self.encode(src_ids, src_mask)
        cache = self.init_generation_cache(encoder_out, device=device)

        generated = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            logits, cache = self.generate_step(generated[:, -1:], cache)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None or top_p is not None:
                next_token = self._sample_with_filtering(next_logits, top_k, top_p)
            else:
                next_token = next_logits.argmax(dim=-1)

            next_token = next_token.unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)

            finished = finished | (next_token.squeeze(-1) == self.config.eos_token_id)
            if finished.all():
                break

        return generated

    def _sample_with_filtering(
        self,
        logits: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample with top-k and/or top-p filtering."""
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_token

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for both encoder and decoder."""
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.encoder.gradient_checkpointing_disable()
        self.decoder.gradient_checkpointing_disable()

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, d_model={self.config.d_model}, "
            f"encoder_layers={self.config.encoder_layers}, decoder_layers={self.config.decoder_layers}, "
            f"params={self.num_parameters() / 1e6:.1f}M"
        )
