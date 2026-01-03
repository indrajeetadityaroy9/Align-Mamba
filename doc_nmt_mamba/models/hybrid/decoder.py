"""
Hybrid Mamba Decoder.

NEW ARCHITECTURE (from plan):
- Layer 0: HYBRID (Mamba + Cross-Attn) - Contextualized Preamble
- Layers 1-7: Mamba only
- Layer 8: HYBRID - Refresh 1
- Layers 9-15: Mamba only
- Layer 16: HYBRID - Refresh 2
- Layers 17-23: Mamba only

Each HYBRID layer:
    x = x + Mamba(RMSNorm(x))           # Position-aware queries
    x = x + CrossAttn(RMSNorm(x), enc)  # Source-aligned output

Total HYBRID Layers: 3 (at indices [0, 8, 16])
Ratio: 3/24 = 1:8 = 12.5%
"""

import math
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..mamba2.norms import RMSNorm
from .layer_builder import build_decoder_layers, LayerType


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


class HybridMambaDecoder(nn.Module):
    """
    Hybrid decoder with HYBRID blocks at strategic positions.

    NEW ARCHITECTURE:
    - HYBRID blocks at [0, 8, 16] contain both Mamba + Cross-Attention
    - Pure Mamba blocks at remaining positions

    Complexity per generation step:
    - Mamba: O(1) via state caching
    - Cross-attention: O(L_src) at HYBRID layers only

    Total: O(L_src) at 3 layers, O(1) at 21 layers
    Much faster than Transformer's O(L_src + L_tgt) at every layer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 128,
        n_heads: int = 12,
        attention_ratio: float = 0.125,
        cross_attn_every: int = 4,
        hybrid_interval: int = 8,
        use_hybrid_blocks: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_layers: Number of decoder layers (default 24 for [0,8,16] pattern)
            d_state: Mamba state dimension
            n_heads: Number of attention heads
            attention_ratio: Fraction of self-attention layers (deprecated)
            cross_attn_every: Deprecated - use hybrid_interval instead
            hybrid_interval: Interval between HYBRID blocks (default 8)
            use_hybrid_blocks: If True, use new HYBRID architecture
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            pad_token_id: Padding token ID
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_token_id = pad_token_id
        self.hybrid_interval = hybrid_interval
        self.use_hybrid_blocks = use_hybrid_blocks
        self.dtype = dtype  # Store for embedding output conversion

        # For state management during inference
        self.expand = 2  # Mamba expansion factor
        self.d_conv = 4  # Mamba conv kernel size

        factory_kwargs = {"device": device, "dtype": dtype}

        # Token embedding (dtype not supported for embedding, only device)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, device=device)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Build hybrid layers
        self.layers, self.layer_types = build_decoder_layers(
            n_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            n_heads=n_heads,
            attention_ratio=attention_ratio,
            cross_attn_every=cross_attn_every,
            hybrid_interval=hybrid_interval,
            use_hybrid_blocks=use_hybrid_blocks,
            dropout=dropout,
            max_seq_len=max_seq_len,
            **factory_kwargs,
        )

        # Final normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Output projection (language model head)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Gradient checkpointing flag
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
        # Embed tokens and convert to model dtype (embeddings output float32)
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        x = self.embed_dropout(x)

        # Apply hybrid layers
        for layer, layer_type in zip(self.layers, self.layer_types):
            if self._gradient_checkpointing and self.training:
                if layer_type == LayerType.HYBRID:
                    # HYBRID: Mamba + Cross-Attention in same block
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, encoder_out, use_reentrant=False
                    )
                elif layer_type == LayerType.CROSS_ATTENTION:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, encoder_out, use_reentrant=False
                    )
                elif layer_type == LayerType.ATTENTION:
                    x, _ = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
            else:
                if layer_type == LayerType.HYBRID:
                    # HYBRID: Mamba + Cross-Attention in same block
                    x = layer(x, encoder_out)
                elif layer_type == LayerType.CROSS_ATTENTION:
                    x = layer(x, encoder_out)
                elif layer_type == LayerType.ATTENTION:
                    x, _ = layer(x)  # Ignore KV cache during training
                else:
                    x = layer(x)

        # Final norm and LM head
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

        CRITICAL: This must be structured correctly from Day 1!
        Mamba layers have fixed-size state, attention has growing KV cache.
        HYBRID layers need Mamba state (cross-attention uses cached encoder).

        Args:
            batch_size: Batch size
            encoder_out: Encoder output (cached for cross-attention)
            device: Device for cache tensors
            dtype: Data type for cache tensors

        Returns:
            Cache dictionary with:
            - ssm_states: Dict[layer_idx, MambaState]
            - kv_caches: Dict[layer_idx, AttentionKVCache]
            - encoder_output: Cached encoder output
            - seqlen_offset: Current position in generation
        """
        device = device or next(self.parameters()).device
        dtype = dtype or torch.bfloat16

        ssm_states = {}
        kv_caches = {}

        layer_idx = 0
        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type == LayerType.MAMBA:
                # Use the wrapper's mamba to allocate correct cache shapes
                conv_state, ssm_state = layer.mamba.allocate_inference_cache(
                    batch_size=batch_size,
                    max_seqlen=1,  # For step-by-step generation
                    dtype=dtype,
                    device=device,
                )
                ssm_states[layer_idx] = MambaState(conv_state, ssm_state)
            elif layer_type == LayerType.HYBRID:
                # HYBRID: need Mamba state (cross-attention uses encoder_output)
                conv_state, ssm_state = layer.mamba.mamba.allocate_inference_cache(
                    batch_size=batch_size,
                    max_seqlen=1,
                    dtype=dtype,
                    device=device,
                )
                ssm_states[layer_idx] = MambaState(conv_state, ssm_state)
            elif layer_type == LayerType.ATTENTION:
                kv_caches[layer_idx] = AttentionKVCache(None, None)
            # CROSS_ATTENTION doesn't need cache (uses encoder_output)
            layer_idx += 1

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
            - logits: (batch, 1, vocab_size)
            - updated_cache: Cache with updated states
        """
        # Embed token and convert to model dtype
        x = self.embed(input_ids) * self.embed_scale  # (B, 1, D)
        if self.dtype is not None:
            x = x.to(self.dtype)

        offset = cache["seqlen_offset"]

        layer_idx = 0
        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type == LayerType.MAMBA:
                # Mamba: pass inference state
                state = cache["ssm_states"].get(layer_idx)
                if state is not None:
                    x = layer(x, inference_params=(state.conv_state, state.ssm_state))
                else:
                    x = layer(x)
            elif layer_type == LayerType.HYBRID:
                # HYBRID: Mamba with state + Cross-attention with cached encoder
                state = cache["ssm_states"].get(layer_idx)
                inference_params = (state.conv_state, state.ssm_state) if state else None
                x = layer(
                    x,
                    cache["encoder_output"],
                    decoder_offset=offset,
                    inference_params=inference_params,
                )
            elif layer_type == LayerType.ATTENTION:
                # Self-attention: update KV cache
                kv_cache = cache["kv_caches"][layer_idx]
                x, new_kv = layer(x, kv_cache=(kv_cache.key_cache, kv_cache.value_cache), offset=offset)
                cache["kv_caches"][layer_idx] = AttentionKVCache(new_kv[0], new_kv[1])
            elif layer_type == LayerType.CROSS_ATTENTION:
                # Cross-attention: use cached encoder output
                x = layer(x, cache["encoder_output"], decoder_offset=offset)
            layer_idx += 1

        # Update offset
        cache["seqlen_offset"] = offset + 1

        # Final norm and LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, cache

    def get_layer_counts(self) -> dict:
        """Get count of each layer type."""
        from .layer_builder import count_layer_types
        return count_layer_types(self.layer_types)

    def extra_repr(self) -> str:
        counts = self.get_layer_counts()
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"mamba={counts['mamba']}, attention={counts['attention']}, "
            f"cross_attention={counts['cross_attention']}"
        )
