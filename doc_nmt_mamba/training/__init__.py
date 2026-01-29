"""Training infrastructure for Document-Level NMT."""

from .objectives import LabelSmoothingCrossEntropy, create_loss_fn, create_scheduler
from .trainer import NMTTrainer, NMTTrainerConfig
from .distributed import setup_distributed, cleanup_distributed
from .hardware import detect_hardware, setup_training_environment

__all__ = [
    "LabelSmoothingCrossEntropy",
    "create_loss_fn",
    "create_scheduler",
    "NMTTrainer",
    "NMTTrainerConfig",
    "setup_distributed",
    "cleanup_distributed",
    "detect_hardware",
    "setup_training_environment",
]
