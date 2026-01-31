"""Model factory for creating models from configuration."""

import torch
from omegaconf import DictConfig

from align_mamba.models.encoder_decoder import HybridMambaEncoderDecoder
from align_mamba.config import ModelConfig, SOTAConfig


def create_model(
    cfg: DictConfig,
    device: str,
    dtype: torch.dtype,
) -> HybridMambaEncoderDecoder:
    """Instantiate model from Hydra config."""
    hybrid_positions = list(cfg.model.hybrid_positions) if cfg.model.hybrid_positions else None

    model_cfg = ModelConfig(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        encoder_layers=cfg.model.encoder_layers,
        decoder_layers=cfg.model.decoder_layers,
        d_state=cfg.model.d_state,
        n_heads=cfg.model.n_heads,
        hybrid_positions=hybrid_positions,
        num_pairs=cfg.data.num_pairs,
        num_samples=cfg.data.num_samples,
        sota=SOTAConfig.from_hydra(cfg),
    )

    return HybridMambaEncoderDecoder(config=model_cfg, device=device, dtype=dtype)
