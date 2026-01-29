"""Encoder-Decoder wrapper and ModelConfig."""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn

from .align_mamba import (
    HybridBiMambaEncoder,
    HybridMambaDecoder,
)
from ..constants import (
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
    DROPOUT, MAX_SEQ_LEN, MQAR_VOCAB_SIZE,
)


@dataclass
class ModelConfig:
    """
    Configuration for the Hybrid Mamba-Attention model.

    Simplified to essential parameters only. Infrastructure values
    (dropout, max_seq_len, token IDs) use constants.
    """
    vocab_size: int = MQAR_VOCAB_SIZE
    d_model: int = 256
    encoder_layers: int = 2
    decoder_layers: int = 4
    d_state: int = 64
    n_heads: int = 8
    hybrid_positions: Optional[List[int]] = None  # None = scale-invariant formula

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class HybridMambaEncoderDecoder(nn.Module):
    """
    Full Encoder-Decoder with Hybrid Mamba-Attention architecture.

    Infrastructure values (dropout, max_seq_len, token IDs) are hardcoded
    from constants.py to reduce configuration complexity.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        vocab_size: int = MQAR_VOCAB_SIZE,
        d_model: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 4,
        d_state: int = 64,
        n_heads: int = 8,
        hybrid_positions: Optional[List[int]] = None,
        share_embeddings: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if config is not None:
            vocab_size = config.vocab_size
            d_model = config.d_model
            encoder_layers = config.encoder_layers
            decoder_layers = config.decoder_layers
            d_state = config.d_state
            n_heads = config.n_heads
            hybrid_positions = config.hybrid_positions

        # Store config for checkpoint serialization
        self.config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            d_state=d_state,
            n_heads=n_heads,
            hybrid_positions=hybrid_positions,
        )

        factory_kwargs = {"device": device, "dtype": dtype}

        self.encoder = HybridBiMambaEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=encoder_layers,
            d_state=d_state,
            n_heads=n_heads,
            dropout=DROPOUT,
            max_seq_len=MAX_SEQ_LEN,
            pad_token_id=PAD_TOKEN_ID,
            **factory_kwargs,
        )

        self.decoder = HybridMambaDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=decoder_layers,
            d_state=d_state,
            n_heads=n_heads,
            hybrid_positions=hybrid_positions,
            dropout=DROPOUT,
            max_seq_len=MAX_SEQ_LEN,
            pad_token_id=PAD_TOKEN_ID,
            **factory_kwargs,
        )

        if share_embeddings:
            self.decoder.embed.weight = self.encoder.embed.weight

    def forward(
        self,
        src_ids: Optional[torch.Tensor],
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. Returns logits (batch, tgt_len, vocab_size)."""
        if src_ids is not None and src_ids.size(1) > MAX_SEQ_LEN:
            raise ValueError(
                f"Source sequence length ({src_ids.size(1)}) exceeds max_seq_len ({MAX_SEQ_LEN})."
            )
        if tgt_ids.size(1) > MAX_SEQ_LEN:
            raise ValueError(
                f"Target sequence length ({tgt_ids.size(1)}) exceeds max_seq_len ({MAX_SEQ_LEN})."
            )

        # Decoder-only mode when src_ids is None or encoder has 0 layers
        if src_ids is None or self.config.encoder_layers == 0:
            encoder_out = None
            encoder_padding_mask = None
        else:
            encoder_out = self.encoder(src_ids, attention_mask=src_mask)
            encoder_padding_mask = src_mask

        logits = self.decoder(
            tgt_ids,
            encoder_out,
            attention_mask=tgt_mask,
            encoder_padding_mask=encoder_padding_mask,
        )

        # Shape invariants
        assert logits.shape[:2] == tgt_ids.shape[:2], (
            f"Logits shape {logits.shape[:2]} != tgt shape {tgt_ids.shape[:2]}"
        )
        assert logits.shape[2] == self.config.vocab_size, (
            f"Logits vocab dim {logits.shape[2]} != config {self.config.vocab_size}"
        )

        return logits

    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(src_ids, attention_mask=src_mask)

    def init_generation_cache(
        self,
        encoder_out: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict:
        batch_size = encoder_out.size(0)
        return self.decoder.init_cache(batch_size, encoder_out, device, dtype)

    def generate_step(
        self,
        input_ids: torch.Tensor,
        cache: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
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
        """Autoregressive generation. Returns (batch, gen_len)."""
        if src_ids.size(1) > MAX_SEQ_LEN:
            raise ValueError(
                f"Source sequence length ({src_ids.size(1)}) exceeds max_seq_len ({MAX_SEQ_LEN})."
            )

        batch_size = src_ids.size(0)
        device = src_ids.device

        encoder_out = self.encode(src_ids, src_mask)
        cache = self.init_generation_cache(encoder_out, device=device)

        generated = torch.full(
            (batch_size, 1),
            BOS_TOKEN_ID,
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

            finished = finished | (next_token.squeeze(-1) == EOS_TOKEN_ID)
            if finished.all():
                break

        return generated

    def _sample_with_filtering(
        self,
        logits: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
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
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()
        self.decoder.gradient_checkpointing_disable()

    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, d_model={self.config.d_model}, "
            f"encoder_layers={self.config.encoder_layers}, decoder_layers={self.config.decoder_layers}, "
            f"params={self.num_parameters() / 1e6:.1f}M"
        )
