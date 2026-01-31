"""Hardware and training infrastructure settings."""

from typing import Tuple

# Optimizer (Chinchilla recipe, arXiv 2203.15556)
ADAM_BETAS: Tuple[float, float] = (0.9, 0.95)
ADAM_EPS: float = 1e-8

# Hardware (H100 optimized)
USE_BF16: bool = True
USE_COMPILE: bool = True
COMPILE_MODE: str = "reduce-overhead"
GRADIENT_CHECKPOINTING: bool = True

# Warmup floor (arXiv 1706.02677)
MIN_WARMUP_STEPS: int = 100
