"""Constants for Align-Mamba experiments.

These values are hardcoded because they never change across experiments.
Keeping them here reduces configuration noise and prevents accidental changes.
"""

# =============================================================================
# Mamba Block Parameters (standard values from mamba-ssm)
# =============================================================================
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2

# =============================================================================
# Token IDs (standard NLP conventions)
# =============================================================================
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3  # Key-value separator ':'
QUERY_TOKEN_ID = 4

# =============================================================================
# MQAR Token Ranges (prevent key/value collision)
# =============================================================================
KEY_TOKEN_START = 10
KEY_TOKEN_END = 4096
VALUE_TOKEN_START = 4096
VALUE_TOKEN_END = 8192

# =============================================================================
# AdamW Optimizer Defaults
# =============================================================================
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8

# =============================================================================
# Training Infrastructure
# =============================================================================
LOG_STEPS = 100
EVAL_STEPS = 1000
SAVE_STEPS = 5000
MIN_LR = 1e-6
MIN_WARMUP_STEPS = 100

# =============================================================================
# Data Loading (tuned for 26-core CPU + H100)
# =============================================================================
NUM_WORKERS = 20
PREFETCH_FACTOR = 8

# =============================================================================
# H100 Training Infrastructure (hardcoded - never changes)
# =============================================================================
USE_BF16 = True
USE_COMPILE = True
COMPILE_MODE = "max-autotune"
USE_AGC = True
AGC_CLIP_FACTOR = 0.01
GRADIENT_CHECKPOINTING = True
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
MAX_SEQ_LEN = 8192

# =============================================================================
# MQAR Task Defaults
# =============================================================================
MQAR_VOCAB_SIZE = 8192
MQAR_SEQ_LENGTH = 512
