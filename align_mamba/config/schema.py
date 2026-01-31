"""Validated configuration schema with single source of defaults."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

from .tokens import MQAR_VOCAB_SIZE


class BlockType(str, Enum):
    """Decoder block types."""
    POLARIZED = "polarized"
    STATE_EXPANDED = "state_expanded"
    MEMMAMBA = "memmamba"


class AttentionType(str, Enum):
    """Cross-attention types."""
    SOFTMAX = "softmax"
    BASED = "based"


@dataclass
class SOTAConfig:
    """SOTA feature configuration."""

    block_type: BlockType = BlockType.POLARIZED
    state_expansion_head_dim: int = 128

    attention_type: AttentionType = AttentionType.SOFTMAX
    based_feature_dim: int = 16
    based_window_size: int = 64

    memmamba_pool_size: int = 50
    memmamba_summary_dim: int = 64
    memmamba_tau1: float = 0.5
    memmamba_tau2: float = 0.3
    memmamba_cross_layer_freq: int = 4

    @classmethod
    def from_hydra(cls, cfg: "DictConfig") -> "SOTAConfig":
        """Build SOTAConfig from Hydra config."""
        sota_cfg = cfg.model.sota
        if not sota_cfg:
            return cls()

        return cls(
            block_type=BlockType(sota_cfg.block_type),
            attention_type=AttentionType(sota_cfg.attention_type),
            state_expansion_head_dim=sota_cfg.state_expansion_head_dim,
            based_feature_dim=sota_cfg.based_feature_dim,
            based_window_size=sota_cfg.based_window_size,
            memmamba_pool_size=sota_cfg.memmamba_pool_size,
            memmamba_summary_dim=sota_cfg.memmamba_summary_dim,
            memmamba_tau1=sota_cfg.memmamba_tau1,
            memmamba_tau2=sota_cfg.memmamba_tau2,
            memmamba_cross_layer_freq=sota_cfg.memmamba_cross_layer_freq,
        )


@dataclass
class ModelConfig:
    """Hybrid Mamba-Attention model configuration."""

    vocab_size: int = MQAR_VOCAB_SIZE
    d_model: int = 256
    encoder_layers: int = 2
    decoder_layers: int = 4
    d_state: int = 64
    n_heads: int = 8
    hybrid_positions: Optional[List[int]] = None
    num_pairs: Optional[int] = None
    num_samples: Optional[int] = None
    sota: SOTAConfig = field(default_factory=SOTAConfig)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
