"""
Encoder implementations for NMT: BiGRU (baseline), Mamba-2 SSM, and Transformer.

This module consolidates encoder configuration, base class, implementations,
and factory into a single file for clarity.
"""

import math
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Configuration ====================

@dataclass
class EncoderConfig:
    """Configuration for encoder models."""
    encoder_type: Literal["bigru", "mamba2", "transformer"] = "bigru"
    vocab_size: int = 50000
    embedding_dim: int = 512
    hidden_dim: int = 1024
    padding_idx: int = 0
    # Mamba-2 specific parameters
    mamba_num_layers: int = 4
    mamba_d_state: int = 64
    mamba_expand: int = 2
    # Transformer specific parameters
    transformer_num_layers: int = 6
    transformer_num_heads: int = 8
    transformer_d_ff: int = 2048
    transformer_dropout: float = 0.1
    transformer_max_seq_len: int = 100


# ==================== Base Class ====================

class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for encoders.

    All encoder implementations return:
    - encoder_outputs: (batch, seq_len, hidden_dim * 2)
    - encoder_hidden: (1, batch, hidden_dim) for decoder initialization
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, padding_idx: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

    @abstractmethod
    def forward(self, input_tokens: torch.Tensor, src_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input tokens."""
        pass


# ==================== GRU Encoder ====================

class GRUEncoder(BaseEncoder):
    """Bidirectional GRU encoder (baseline)."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, padding_idx: int):
        super().__init__(vocab_size, embedding_dim, hidden_dim, padding_idx)

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.bi_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.hidden_fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input_tokens: torch.Tensor, src_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_tokens = self.embedding_layer(input_tokens)
        gru_outputs, gru_hidden = self.bi_gru(embedded_tokens)

        # gru_hidden: (2, batch, hidden_dim) for bidirectional
        forward_hidden = gru_hidden[0, :, :]
        backward_hidden = gru_hidden[1, :, :]

        encoder_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        encoder_hidden = torch.tanh(self.hidden_fc(encoder_hidden))
        encoder_hidden = encoder_hidden.unsqueeze(0)

        return gru_outputs, encoder_hidden


# ==================== Mamba-2 Encoder ====================

def _length_aware_flip(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Flip sequences respecting actual lengths (not padding).

    CRITICAL: Naive .flip(1) breaks with padded sequences:
    [A, B, C, <pad>, <pad>].flip(1) = [<pad>, <pad>, C, B, A]  # WRONG!

    This correctly flips only the non-padded portion:
    [A, B, C, <pad>, <pad>] -> [C, B, A, <pad>, <pad>]  # CORRECT!
    """
    batch_size, seq_len, hidden_dim = x.shape
    x_flipped = x.clone()

    for i in range(batch_size):
        length = lengths[i].item()
        if length > 0:
            x_flipped[i, :length] = x[i, :length].flip(0)

    return x_flipped


class _BidirectionalMamba2Block(nn.Module):
    """
    Bidirectional Mamba-2 block using forward + backward passes.

    Since Mamba is inherently unidirectional, we simulate bidirectionality
    by running two separate Mamba blocks and concatenating outputs.
    """

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2):
        super().__init__()
        from mamba_ssm import Mamba2

        self.d_model = d_model
        self.mamba_forward = Mamba2(d_model=d_model, d_state=d_state, expand=expand)
        self.mamba_backward = Mamba2(d_model=d_model, d_state=d_state, expand=expand)
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        forward_out = self.mamba_forward(x)

        x_flipped = _length_aware_flip(x, lengths)
        backward_out_flipped = self.mamba_backward(x_flipped)
        backward_out = _length_aware_flip(backward_out_flipped, lengths)

        combined = torch.cat([forward_out, backward_out], dim=-1)
        output = self.output_proj(combined)

        return self.layer_norm(output + x)


class MambaEncoder(BaseEncoder):
    """
    Bidirectional Mamba-2 encoder for NMT.

    Uses stacked BidirectionalMamba2Blocks with mean pooling + adapter
    for decoder hidden state initialization.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        padding_idx: int,
        num_layers: int = 4,
        d_state: int = 64,
        expand: int = 2
    ):
        super().__init__(vocab_size, embedding_dim, hidden_dim, padding_idx)

        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        self.mamba_layers = nn.ModuleList([
            _BidirectionalMamba2Block(d_model=hidden_dim, d_state=d_state, expand=expand)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.adapter = nn.Linear(hidden_dim * 2, hidden_dim)

        nn.init.xavier_uniform_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, input_tokens: torch.Tensor, src_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_tokens.shape

        if src_lengths is None:
            src_lengths = (input_tokens != self.padding_idx).sum(dim=1)

        embedded = self.embedding_layer(input_tokens)
        hidden = self.input_proj(embedded)

        for mamba_layer in self.mamba_layers:
            hidden = mamba_layer(hidden, src_lengths)

        encoder_outputs = self.output_proj(hidden)

        # Mean pooling for decoder initialization (excluding padding)
        mask = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()

        pooled = (encoder_outputs * mask).sum(dim=1) / src_lengths.unsqueeze(1).float().clamp(min=1)
        encoder_hidden = torch.tanh(self.adapter(pooled))
        encoder_hidden = encoder_hidden.unsqueeze(0)

        return encoder_outputs, encoder_hidden


# ==================== Transformer Encoder ====================

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for Transformer.

    Uses learned embeddings (more effective for shorter sequences like Multi30k)
    with optional sinusoidal initialization.
    """

    def __init__(self, d_model: int, max_seq_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Initialize with sinusoidal pattern for better starting point
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_embedding.weight.data.copy_(pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input embeddings
        Returns:
            (batch, seq_len, d_model) embeddings with positional encoding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_enc = self.pos_embedding(positions)
        return self.dropout(x + pos_enc)


class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer with GELU activation.

    Pre-norm architecture is more stable for training compared to post-norm.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input
            src_key_padding_mask: (batch, seq_len) True for padding positions
        Returns:
            (batch, seq_len, d_model) output
        """
        # Pre-norm self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)

        # Pre-norm FFN with residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class TransformerEncoder(BaseEncoder):
    """
    Transformer encoder for NMT with pre-norm architecture.

    Features:
    - Learned positional encodings (initialized with sinusoidal)
    - Pre-norm for stable training
    - GELU activation in FFN
    - Compatible output interface with BiGRU/Mamba encoders

    Output:
    - encoder_outputs: (batch, seq_len, hidden_dim * 2) for decoder attention
    - encoder_hidden: (1, batch, hidden_dim) for decoder initialization
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        padding_idx: int,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__(vocab_size, embedding_dim, hidden_dim, padding_idx)

        self.num_layers = num_layers
        self.d_model = embedding_dim

        # Embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embed_scale = math.sqrt(embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Output projection to match decoder expectation (hidden_dim * 2)
        self.output_proj = nn.Linear(embedding_dim, hidden_dim * 2)

        # Adapter for decoder hidden state initialization
        self.hidden_adapter = nn.Linear(embedding_dim, hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_tokens: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input tokens.

        Args:
            input_tokens: (batch, seq_len) input token IDs
            src_lengths: (batch,) actual lengths (used for padding mask)

        Returns:
            encoder_outputs: (batch, seq_len, hidden_dim * 2)
            encoder_hidden: (1, batch, hidden_dim)
        """
        batch_size, seq_len = input_tokens.shape

        # Create padding mask
        if src_lengths is not None:
            # True for padding positions (to be masked)
            src_key_padding_mask = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0) >= src_lengths.unsqueeze(1)
        else:
            src_key_padding_mask = (input_tokens == self.padding_idx)

        # Embedding + positional encoding
        x = self.embedding(input_tokens) * self.embed_scale
        x = self.pos_encoding(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        # Final normalization
        x = self.final_norm(x)

        # Project to decoder dimension
        encoder_outputs = self.output_proj(x)

        # Mean pooling for decoder initialization (excluding padding)
        if src_lengths is not None:
            mask = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0) < src_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / src_lengths.unsqueeze(1).float().clamp(min=1)
        else:
            # No lengths provided, use simple mean
            pooled = x.mean(dim=1)

        encoder_hidden = torch.tanh(self.hidden_adapter(pooled))
        encoder_hidden = encoder_hidden.unsqueeze(0)

        return encoder_outputs, encoder_hidden


# ==================== Factory ====================

def create_encoder(config: EncoderConfig) -> BaseEncoder:
    """Create an encoder based on configuration."""
    if config.encoder_type == "bigru":
        return GRUEncoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            padding_idx=config.padding_idx
        )
    elif config.encoder_type == "mamba2":
        return MambaEncoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            padding_idx=config.padding_idx,
            num_layers=config.mamba_num_layers,
            d_state=config.mamba_d_state,
            expand=config.mamba_expand
        )
    elif config.encoder_type == "transformer":
        return TransformerEncoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            padding_idx=config.padding_idx,
            num_layers=config.transformer_num_layers,
            num_heads=config.transformer_num_heads,
            d_ff=config.transformer_d_ff,
            dropout=config.transformer_dropout,
            max_seq_len=config.transformer_max_seq_len
        )
    else:
        raise ValueError(f"Unknown encoder type: {config.encoder_type}. Use 'bigru', 'mamba2', or 'transformer'.")
