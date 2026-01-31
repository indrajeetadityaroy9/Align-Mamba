"""Unified configuration system for Align-Mamba."""

from .schema import BlockType, AttentionType, SOTAConfig, ModelConfig
from .tokens import (
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID, QUERY_TOKEN_ID,
    KEY_TOKEN_START, KEY_TOKEN_END, VALUE_TOKEN_START, VALUE_TOKEN_END,
    MQAR_VOCAB_SIZE, MQAR_SEQ_LENGTH, MAX_SEQ_LEN,
)
from .hardware import (
    ADAM_BETAS, ADAM_EPS,
    USE_BF16, USE_COMPILE, COMPILE_MODE, GRADIENT_CHECKPOINTING,
    MIN_WARMUP_STEPS,
)

__all__ = [
    # Schema
    "BlockType", "AttentionType", "SOTAConfig", "ModelConfig",
    # Tokens
    "PAD_TOKEN_ID", "BOS_TOKEN_ID", "EOS_TOKEN_ID", "SEP_TOKEN_ID", "QUERY_TOKEN_ID",
    "KEY_TOKEN_START", "KEY_TOKEN_END", "VALUE_TOKEN_START", "VALUE_TOKEN_END",
    "MQAR_VOCAB_SIZE", "MQAR_SEQ_LENGTH", "MAX_SEQ_LEN",
    # Hardware
    "ADAM_BETAS", "ADAM_EPS",
    "USE_BF16", "USE_COMPILE", "COMPILE_MODE", "GRADIENT_CHECKPOINTING",
    "MIN_WARMUP_STEPS",
]
