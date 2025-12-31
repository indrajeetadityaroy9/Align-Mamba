"""Training utilities for NMT."""

from training.trainer import train_model
from training.evaluator import validate
from training.orpo_loss import ORPOLoss
from training.orpo_trainer import train_orpo
from training.hardware import (
    configure_hardware,
    get_device_info,
    get_optimal_batch_size,
    get_optimal_workers,
    compile_model,
    create_optimized_dataloader,
    GradientAccumulator,
)
from training.label_smoothing import LabelSmoothingLoss, LabelSmoothingCrossEntropy
from training.lr_scheduler import (
    WarmupCosineScheduler,
    WarmupInverseSquareRootScheduler,
    create_scheduler,
)
from training.rdrop import RDropLoss, compute_kl_divergence, rdrop_forward_pass
from training.beam_search import (
    BeamSearchConfig,
    beam_search_decode,
    beam_search_validate,
)
from training.enhanced_trainer import train_enhanced

__all__ = [
    'train_model',
    'validate',
    'ORPOLoss',
    'train_orpo',
    'configure_hardware',
    'get_device_info',
    'get_optimal_batch_size',
    'get_optimal_workers',
    'compile_model',
    'create_optimized_dataloader',
    'GradientAccumulator',
    # Label smoothing
    'LabelSmoothingLoss',
    'LabelSmoothingCrossEntropy',
    # LR schedulers
    'WarmupCosineScheduler',
    'WarmupInverseSquareRootScheduler',
    'create_scheduler',
    # R-Drop
    'RDropLoss',
    'compute_kl_divergence',
    'rdrop_forward_pass',
    # Beam search
    'BeamSearchConfig',
    'beam_search_decode',
    'beam_search_validate',
    # Enhanced trainer
    'train_enhanced',
]
