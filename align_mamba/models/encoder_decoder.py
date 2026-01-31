"""Encoder-Decoder wrapper and checkpoint utilities."""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from align_mamba.models.encoder import HybridBiMambaEncoder
from align_mamba.models.decoder import HybridMambaDecoder
from align_mamba.config import PAD_TOKEN_ID, MAX_SEQ_LEN, ModelConfig
from align_mamba.training.optimization import compute_adaptive_dropout


class HybridMambaEncoderDecoder(nn.Module):
    """Full Encoder-Decoder with Hybrid Mamba-Attention architecture."""

    def __init__(
        self,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}

        estimated_params = (
            config.vocab_size * config.d_model * 2 +
            config.encoder_layers * config.d_model * config.d_model * 4 +
            config.decoder_layers * config.d_model * config.d_model * 4
        )
        num_samples = config.num_samples if config.num_samples else 100000
        dropout = compute_adaptive_dropout(estimated_params, num_samples)

        self.encoder = HybridBiMambaEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.encoder_layers,
            d_state=config.d_state,
            n_heads=config.n_heads,
            dropout=dropout,
            max_seq_len=MAX_SEQ_LEN,
            pad_token_id=PAD_TOKEN_ID,
            **factory_kwargs,
        )

        self.decoder = HybridMambaDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.decoder_layers,
            d_state=config.d_state,
            n_heads=config.n_heads,
            hybrid_positions=config.hybrid_positions,
            num_pairs=config.num_pairs,
            dropout=dropout,
            max_seq_len=MAX_SEQ_LEN,
            pad_token_id=PAD_TOKEN_ID,
            sota_config=config.sota,
            **factory_kwargs,
        )

    def forward(
        self,
        src_ids: Optional[torch.Tensor],
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. Returns logits (batch, tgt_len, vocab_size)."""
        if src_ids is None or self.config.encoder_layers == 0:
            encoder_out = None
            encoder_padding_mask = None
        else:
            encoder_out = self.encoder(src_ids, attention_mask=src_mask)
            encoder_padding_mask = src_mask

        return self.decoder(
            tgt_ids,
            encoder_out,
            attention_mask=tgt_mask,
            encoder_padding_mask=encoder_padding_mask,
        )

    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """Unwrap model from torch.compile and DDP wrappers."""
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def load_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, ModelConfig]:
    """Load model from checkpoint for evaluation."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ModelConfig(**checkpoint['config'])

    model = HybridMambaEncoderDecoder(config=config, device=device, dtype=dtype)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config
