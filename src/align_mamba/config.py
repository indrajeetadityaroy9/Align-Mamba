# Configuration and YAML loading.
import warnings
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

# MQAR token protocol.
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
    # Model, data, and runtime parameters.
    d_model: int = 256
    encoder_layers: int = 6
    decoder_layers: int = 6
    d_state: int = 64
    n_heads: int = 4
    encoder_inject_layers: tuple = (0, 2)
    encoder_attn_layers: tuple = (3, 5)
    n_persistent_mem: int = 8
    block_size: int = 4
    n_householder_steps: int = 2
    kronecker_partitions: int = 5
    kronecker_subdim: int = 4
    top_k_slots: int = 8
    decoder_attn_layer: int = 4

    decay_gamma_init: float = 1.0
    surprise_eta_init: float = 0.9

    batch_size: int = 256
    max_steps: int = 100000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    num_pairs: int = 128
    num_queries: int = 16
    num_samples: int = 100000
    val_ratio: float = 0.1

    seed: int = 42
    output_dir: str = "results"
    dtype: str = "bfloat16"


_TUPLE_FIELDS = frozenset(
    f.name for f in fields(Config) if f.type in (tuple, "tuple")
)


def _deep_merge(base: dict, override: dict) -> dict:
    # Recursively merge override into base (override wins).
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_yaml(path: str):
    # Load YAML config with optional `base` inheritance.
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "base" in raw:
        base_path = str(Path(path).parent / raw.pop("base"))
        with open(base_path) as bf:
            base_raw = yaml.safe_load(bf)
        raw = _deep_merge(base_raw, raw)
    flat = {}
    for section in ('run', 'model', 'data', 'training'):
        flat.update(raw[section])
    valid = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in flat.items() if k in valid}
    unknown = set(flat) - valid
    if unknown:
        warnings.warn(f"Unknown config keys (ignored): {unknown}")
    for name in _TUPLE_FIELDS:
        if name in filtered and isinstance(filtered[name], list):
            filtered[name] = tuple(filtered[name])
    return Config(**filtered), raw
