"""
Hybrid BiMamba Encoder.

Combines:
- BiMamba blocks for bidirectional context (O(L) complexity)
- Sparse bidirectional attention for in-context learning (1:7 ratio)
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn

from ..mamba2.norms import RMSNorm
from .layer_builder import build_encoder_layers, LayerType


class HybridBiMambaEncoder(nn.Module):
    """
    Hybrid encoder with BiMamba + sparse bidirectional attention.

    BiMamba provides bidirectional context with O(L) complexity.
    Strategic attention layers (1:7 ratio) enable in-context learning.
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
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_layers: Number of encoder layers
            d_state: Mamba state dimension
            n_heads: Number of attention heads
            attention_ratio: Fraction of attention layers (1/8 = 1:7 ratio)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            pad_token_id: Padding token ID
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_token_id = pad_token_id
        self.dtype = dtype  # Store for embedding output conversion

        factory_kwargs = {"device": device, "dtype": dtype}

        # Token embedding (dtype not supported for embedding, only device)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id, device=device)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Build hybrid layers
        self.layers, self.layer_types = build_encoder_layers(
            n_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            n_heads=n_heads,
            attention_ratio=attention_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
            **factory_kwargs,
        )

        # Final normalization
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input sequence.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask (batch, seq_len).
                           True/1 = attend, False/0 = mask (padding).
                           Applied to attention layers to prevent attending to padding.

        Returns:
            Encoder output (batch, seq_len, d_model)
        """
        # Embed tokens and convert to model dtype (embeddings output float32)
        x = self.embed(input_ids) * self.embed_scale
        if self.dtype is not None:
            x = x.to(self.dtype)
        x = self.embed_dropout(x)

        # Apply hybrid layers
        for layer, layer_type in zip(self.layers, self.layer_types):
            if self._gradient_checkpointing and self.training:
                if layer_type == LayerType.ATTENTION:
                    # Pass attention_mask to attention layers
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, attention_mask, use_reentrant=False
                    )
                else:
                    # BiMamba layers don't use attention_mask
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
            else:
                if layer_type == LayerType.ATTENTION:
                    # Pass attention_mask to attention layers
                    x = layer(x, attention_mask=attention_mask)
                else:
                    # BiMamba layers don't use attention_mask
                    x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        return x

    def get_layer_counts(self) -> dict:
        """Get count of each layer type."""
        from .layer_builder import count_layer_types
        return count_layer_types(self.layer_types)

    def extra_repr(self) -> str:
        counts = self.get_layer_counts()
        return (
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"bimamba={counts['bimamba']}, attention={counts['attention']}"
        )
