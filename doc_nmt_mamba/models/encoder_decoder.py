"""
Full Hybrid Mamba-Attention Encoder-Decoder for Document-Level NMT.

This is the main model file that ties together:
- HybridBiMambaEncoder (bidirectional)
- HybridMambaDecoder (causal with cross-attention)
- HybridCacheParams for efficient autoregressive generation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Union, List

import torch
import torch.nn as nn

from .hybrid import HybridBiMambaEncoder, HybridMambaDecoder, MambaState, AttentionKVCache


@dataclass
class ModelConfig:
    """Configuration for the Hybrid Mamba-Attention model.

    NEW ARCHITECTURE (from plan):
    - Decoder uses HYBRID blocks at [0, 8, 16] for 24-layer model
    - Each HYBRID block contains Mamba + Cross-Attention
    - Ratio: 3/24 = 1:8 = 12.5%
    """

    vocab_size: int = 32000
    d_model: int = 768
    encoder_layers: int = 16
    decoder_layers: int = 24  # Changed to 24 for [0,8,16] pattern
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 12
    attention_ratio: float = 0.125  # 1:7 ratio for encoder
    cross_attn_every: int = 4  # Deprecated - use hybrid_interval
    hybrid_interval: int = 8  # NEW: interval between HYBRID blocks
    use_hybrid_blocks: bool = True  # NEW: use HYBRID block architecture
    dropout: float = 0.1
    max_seq_len: int = 8192
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class HybridCacheParams:
    """
    Hybrid cache for Mamba + Attention layers during autoregressive generation.

    CRITICAL: This structure must be correct from Day 1!
    - Mamba: Fixed-size state (B, d_model*expand, d_state)
    - Attention: Growing KV cache (B, current_len, n_heads, head_dim)

    The cache is managed by the decoder's step() method.
    """

    ssm_states: Dict[int, MambaState] = field(default_factory=dict)
    kv_caches: Dict[int, AttentionKVCache] = field(default_factory=dict)
    encoder_output: Optional[torch.Tensor] = None
    seqlen_offset: int = 0


class HybridMambaEncoderDecoder(nn.Module):
    """
    Full Encoder-Decoder model with Hybrid Mamba-Attention architecture.

    Features:
    - BiMamba encoder with sparse bidirectional attention
    - Causal Mamba decoder with sparse self-attention + cross-attention
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
        cross_attn_every: int = 4,
        hybrid_interval: int = 8,
        use_hybrid_blocks: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        share_embeddings: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            config: Optional ModelConfig (overrides other args if provided)
            vocab_size: Size of vocabulary
            d_model: Model dimension
            encoder_layers: Number of encoder layers
            decoder_layers: Number of decoder layers (24 for [0,8,16] pattern)
            d_state: Mamba state dimension
            n_heads: Number of attention heads
            attention_ratio: Fraction of attention layers (1/8 = 1:7 ratio)
            cross_attn_every: Deprecated - use hybrid_interval
            hybrid_interval: Interval between HYBRID blocks (default 8)
            use_hybrid_blocks: Whether to use HYBRID block architecture
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            share_embeddings: Whether to share encoder/decoder embeddings
            device: Device for parameters
            dtype: Data type for parameters
        """
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
            cross_attn_every = config.cross_attn_every
            hybrid_interval = config.hybrid_interval
            use_hybrid_blocks = config.use_hybrid_blocks
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
            cross_attn_every=cross_attn_every,
            hybrid_interval=hybrid_interval,
            use_hybrid_blocks=use_hybrid_blocks,
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
            attention_ratio=attention_ratio,
            cross_attn_every=cross_attn_every,
            hybrid_interval=hybrid_interval,
            use_hybrid_blocks=use_hybrid_blocks,
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
        # Encode source
        encoder_out = self.encoder(src_ids, attention_mask=src_mask)

        # Decode target
        logits = self.decoder(tgt_ids, encoder_out, attention_mask=tgt_mask)

        return logits

    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode source sequence (for generation).

        Args:
            src_ids: Source token IDs (batch, src_len)
            src_mask: Optional source attention mask

        Returns:
            Encoder output (batch, src_len, d_model)
        """
        return self.encoder(src_ids, attention_mask=src_mask)

    def init_generation_cache(
        self,
        encoder_out: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict:
        """
        Initialize cache for autoregressive generation.

        Args:
            encoder_out: Encoder output (batch, src_len, d_model)
            device: Device for cache tensors
            dtype: Data type for cache tensors

        Returns:
            Cache dictionary for decoder.step()
        """
        batch_size = encoder_out.size(0)
        return self.decoder.init_cache(batch_size, encoder_out, device, dtype)

    def generate_step(
        self,
        input_ids: torch.Tensor,
        cache: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Single generation step.

        Args:
            input_ids: Current token IDs (batch, 1)
            cache: Cache from init_generation_cache or previous step

        Returns:
            Tuple of (logits, updated_cache)
        """
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
            temperature: Sampling temperature (1.0 = no change)
            top_k: Top-k sampling (None = disabled)
            top_p: Top-p (nucleus) sampling (None = disabled)
            src_mask: Optional source attention mask

        Returns:
            Generated token IDs (batch, gen_len)
        """
        batch_size = src_ids.size(0)
        device = src_ids.device

        # Encode source
        encoder_out = self.encode(src_ids, src_mask)

        # Initialize cache
        cache = self.init_generation_cache(encoder_out, device=device)

        # Start with BOS token
        generated = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            # Get next token logits
            logits, cache = self.generate_step(generated[:, -1:], cache)
            next_logits = logits[:, -1, :]  # (batch, vocab)

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Sample next token
            if top_k is not None or top_p is not None:
                next_token = self._sample_with_filtering(next_logits, top_k, top_p)
            else:
                next_token = next_logits.argmax(dim=-1)

            next_token = next_token.unsqueeze(-1)  # (batch, 1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
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
        # Top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        # Sample from filtered distribution
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
