"""Configuration for Align-Mamba."""

from dataclasses import dataclass, asdict, fields

# Token constants (MQAR protocol)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
KEY_TOKEN_START = 10
KEY_TOKEN_END = 4096
VALUE_TOKEN_START = 4096
VALUE_TOKEN_END = 8192
VOCAB_SIZE = 8192
MAX_SEQ_LEN = 8192


@dataclass
class Config:
    """SOTA configuration with principled defaults."""

    # Architecture
    d_model: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 6
    d_state: int = 64
    n_heads: int = 4

    # Cross-attention positions in decoder
    cross_attn_layers: tuple = (0, 2)

    # Memory pool
    mem_pool_size: int = 50
    mem_summary_dim: int = 64
    mem_update_freq: int = 4

    # Training
    batch_size: int = 256
    max_steps: int = 100000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.01

    # Data
    num_pairs: int = 128
    num_queries: int = 16
    num_samples: int = 100000

    # Runtime
    seed: int = 42
    output_dir: str = "outputs"

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def warmup_steps(self) -> int:
        return int(self.warmup_ratio * self.max_steps)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Reconstruct from checkpoint dict."""
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        if 'cross_attn_layers' in filtered and isinstance(filtered['cross_attn_layers'], list):
            filtered['cross_attn_layers'] = tuple(filtered['cross_attn_layers'])
        if 'hybrid_positions' in data:
            filtered['cross_attn_layers'] = tuple(data['hybrid_positions'])
        return cls(**filtered)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['vocab_size'] = self.vocab_size
        d['warmup_steps'] = self.warmup_steps
        return d
