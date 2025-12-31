"""GRU Decoder with Bahdanau Attention for NMT.

Includes:
- Decoder: Original Bahdanau attention decoder (baseline)
- DeepDecoder: Enhanced decoder with multi-head attention, coverage, and deeper layers
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """GRU Decoder with Bahdanau Attention."""

    def __init__(self, hidden_dim: int, vocab_size: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Bahdanau attention layers
        self.attn_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.attn_encoder = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

        # GRU and output projection
        self.gru = nn.GRU(embedding_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_input: torch.Tensor, decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step decoding with attention.

        Args:
            decoder_input: (batch,) current token indices
            decoder_hidden: (1, batch, hidden_dim) current hidden state
            encoder_outputs: (batch, src_len, hidden_dim * 2) encoder outputs

        Returns:
            output_scores: (batch, 1, vocab_size) vocabulary scores
            decoder_hidden: (1, batch, hidden_dim) updated hidden state
        """
        seq_len = encoder_outputs.size(1)
        embedded_input = self.embedding_layer(decoder_input).unsqueeze(1)

        # Attention computation
        decoder_hidden_t = decoder_hidden.permute(1, 0, 2).contiguous()
        decoder_hidden_expanded = decoder_hidden_t.expand(-1, seq_len, -1)

        attn_hidden = self.attn_hidden(decoder_hidden_expanded)
        attn_encoder = self.attn_encoder(encoder_outputs)
        align_scores = torch.tanh(attn_hidden + attn_encoder)
        attn_scores = self.attn_score(align_scores).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        # GRU forward
        gru_input = torch.cat((embedded_input, context_vector), dim=2)
        gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
        output_vocab_scores = self.output_linear(gru_output)

        return output_vocab_scores, decoder_hidden


# ==================== Multi-Head Attention ====================

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with optional coverage mechanism.

    Coverage mechanism prevents the model from attending to the same
    source positions repeatedly, reducing over/under-translation.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_coverage: bool = True
    ):
        super().__init__()
        assert query_dim % num_heads == 0, f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_coverage = use_coverage

        # Linear projections
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        # Coverage projection (maps coverage vector to attention influence)
        if use_coverage:
            self.coverage_proj = nn.Linear(1, num_heads, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.use_coverage:
            nn.init.zeros_(self.coverage_proj.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-head cross-attention with coverage.

        Args:
            query: (batch, 1, query_dim) decoder hidden state
            key: (batch, src_len, key_dim) encoder outputs
            value: (batch, src_len, key_dim) encoder outputs
            coverage: (batch, src_len) accumulated attention (for coverage)
            key_padding_mask: (batch, src_len) True for padding positions

        Returns:
            output: (batch, 1, query_dim) attention output
            attn_weights: (batch, src_len) attention weights (averaged over heads)
            new_coverage: (batch, src_len) updated coverage vector
        """
        batch_size, src_len, _ = key.shape

        # Project queries, keys, values
        q = self.q_proj(query)  # (batch, 1, query_dim)
        k = self.k_proj(key)    # (batch, src_len, query_dim)
        v = self.v_proj(value)  # (batch, src_len, query_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (batch, num_heads, 1, head_dim)
        # k, v: (batch, num_heads, src_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # scores: (batch, num_heads, 1, src_len)

        # Add coverage influence (discourages re-attending to same positions)
        if self.use_coverage and coverage is not None:
            coverage_influence = self.coverage_proj(coverage.unsqueeze(-1))
            # coverage_influence: (batch, src_len, num_heads)
            coverage_influence = coverage_influence.permute(0, 2, 1).unsqueeze(2)
            # coverage_influence: (batch, num_heads, 1, src_len)
            scores = scores - coverage_influence

        # Apply padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (batch, 1, 1, src_len)
                float('-inf')
            )

        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # attn_weights: (batch, num_heads, 1, src_len)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        # context: (batch, num_heads, 1, head_dim)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        output = self.out_proj(context)

        # Average attention weights across heads for coverage update
        avg_attn = attn_weights.mean(dim=1).squeeze(1)  # (batch, src_len)

        # Update coverage
        if coverage is not None:
            new_coverage = coverage + avg_attn
        else:
            new_coverage = avg_attn

        return output, avg_attn, new_coverage


# ==================== Deep Decoder ====================

class DeepDecoder(nn.Module):
    """
    Enhanced GRU Decoder with Multi-Head Attention, Coverage, and Deeper Layers.

    Improvements over baseline Decoder:
    1. Multi-head cross-attention (8 heads) instead of Bahdanau
    2. Coverage mechanism to prevent over/under-translation
    3. Stacked GRU layers (3) with residual connections
    4. Layer normalization for stable training
    5. FFN with GELU activation
    6. Dropout for regularization

    Compatible with both word-level and BPE tokenization.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        embedding_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        output_dropout: float = 0.3,
        use_coverage: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_coverage = use_coverage

        # Embedding with dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embedding_dropout)
        self.embed_scale = math.sqrt(embedding_dim)

        # Input projection (embedding + context -> hidden_dim)
        self.input_proj = nn.Linear(embedding_dim + hidden_dim * 2, hidden_dim)

        # Multi-head cross-attention
        self.cross_attention = MultiHeadCrossAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim * 2,  # encoder outputs are hidden_dim * 2
            num_heads=num_heads,
            dropout=attention_dropout,
            use_coverage=use_coverage
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Stacked GRU layers with residual connections
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_layers)
        ])
        self.gru_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(output_dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(output_dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_dropout = nn.Dropout(output_dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        decoder_input: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step decoding with multi-head attention and coverage.

        Args:
            decoder_input: (batch,) current token indices
            decoder_hidden: (num_layers, batch, hidden_dim) stacked hidden states
            encoder_outputs: (batch, src_len, hidden_dim * 2) encoder outputs
            coverage: (batch, src_len) accumulated attention (optional)
            src_padding_mask: (batch, src_len) True for padding positions (optional)

        Returns:
            output_scores: (batch, 1, vocab_size) vocabulary scores
            decoder_hidden: (num_layers, batch, hidden_dim) updated hidden states
            attn_weights: (batch, src_len) attention weights
            coverage: (batch, src_len) updated coverage vector
        """
        batch_size = decoder_input.size(0)

        # Embedding with scaling and dropout
        embedded = self.embedding(decoder_input) * self.embed_scale
        embedded = self.embed_dropout(embedded).unsqueeze(1)
        # embedded: (batch, 1, embedding_dim)

        # Get query from last layer's hidden state
        query = decoder_hidden[-1:].permute(1, 0, 2)
        # query: (batch, 1, hidden_dim)

        # Multi-head cross-attention
        context, attn_weights, new_coverage = self.cross_attention(
            query=query,
            key=encoder_outputs,
            value=encoder_outputs,
            coverage=coverage,
            key_padding_mask=src_padding_mask
        )
        # context: (batch, 1, hidden_dim)

        # Combine embedding and context
        combined = torch.cat([embedded, context.expand(-1, -1, self.hidden_dim * 2 // self.hidden_dim * self.hidden_dim)], dim=-1)
        # Project to hidden dim
        hidden = self.input_proj(torch.cat([embedded, encoder_outputs.mean(dim=1, keepdim=True).expand(-1, 1, -1)], dim=-1))
        hidden = hidden + context  # Residual connection with attention
        hidden = self.attn_norm(hidden)

        # Stacked GRU layers with residual connections
        new_hiddens = []
        for i, (gru, norm) in enumerate(zip(self.gru_layers, self.gru_norms)):
            layer_hidden = decoder_hidden[i:i+1]  # (1, batch, hidden_dim)
            gru_out, new_h = gru(hidden, layer_hidden)
            # Residual connection + layer norm
            hidden = norm(hidden + gru_out)
            new_hiddens.append(new_h)

        new_decoder_hidden = torch.cat(new_hiddens, dim=0)
        # new_decoder_hidden: (num_layers, batch, hidden_dim)

        # Feed-forward network with residual
        ffn_out = self.ffn(hidden)
        hidden = self.ffn_norm(hidden + ffn_out)

        # Output projection
        output = self.output_dropout(hidden)
        output_scores = self.output_proj(output)
        # output_scores: (batch, 1, vocab_size)

        return output_scores, new_decoder_hidden, attn_weights, new_coverage

    def init_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        Initialize decoder hidden states from encoder final hidden.

        Args:
            encoder_hidden: (1, batch, hidden_dim) from encoder

        Returns:
            decoder_hidden: (num_layers, batch, hidden_dim)
        """
        # Replicate encoder hidden for all decoder layers
        return encoder_hidden.expand(self.num_layers, -1, -1).contiguous()


def compute_coverage_loss(
    attn_weights: torch.Tensor,
    coverage: torch.Tensor,
    src_padding_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute coverage loss to penalize repeated attention.

    Loss = sum(min(attn, coverage)) - encourages diverse attention patterns

    Args:
        attn_weights: (batch, src_len) current attention weights
        coverage: (batch, src_len) accumulated coverage before this step
        src_padding_mask: (batch, src_len) True for padding positions

    Returns:
        coverage_loss: scalar tensor
    """
    # Coverage loss: sum of min(attn, coverage) over non-padded positions
    cov_loss = torch.min(attn_weights, coverage)

    if src_padding_mask is not None:
        cov_loss = cov_loss.masked_fill(src_padding_mask, 0.0)

    return cov_loss.sum() / attn_weights.size(0)  # Average over batch
