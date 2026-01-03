"""
Training infrastructure for Document-Level NMT.

Consolidated structure:
- objectives.py: Loss functions + LR schedulers
- trainer.py: Training loop, evaluation, checkpoints
- distributed.py: DDP/FSDP setup, barrier sync
- hardware.py: H100 optimizations, TF32, memory tracking

Provides:
- LabelSmoothingCrossEntropy for NMT training
- CosineAnnealingWarmupScheduler for LR scheduling
- H100-optimized Trainer with bf16 and torch.compile
- Multi-GPU DDP/FSDP distributed training
- Hardware detection and optimization utilities
"""

from .objectives import (
    # Loss functions
    LabelSmoothingCrossEntropy,
    SequenceLoss,
    PackedSequenceLoss,
    create_loss_fn,
    # Schedulers
    CosineAnnealingWarmupScheduler,
    InverseSqrtScheduler,
    LinearWarmupDecayScheduler,
    PolynomialDecayScheduler,
    create_scheduler,
)
from .trainer import Trainer, TrainerConfig
from .distributed import (
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    wrap_model_distributed,
    get_nvlink_info,
)
from .hardware import (
    HardwareInfo,
    GPUInfo,
    detect_hardware,
    print_hardware_info,
    setup_h100_optimizations,
    setup_nccl_optimizations,
    setup_training_environment,
    CUDAMemoryManager,
    is_ampere_or_newer,
    is_hopper,
    # H100-specific utilities
    get_optimal_worker_count,
    get_available_ram_gb,
    should_preload_dataset,
    print_h100_optimization_status,
)

__all__ = [
    # Loss functions
    "LabelSmoothingCrossEntropy",
    "SequenceLoss",
    "PackedSequenceLoss",
    "create_loss_fn",
    # Schedulers
    "CosineAnnealingWarmupScheduler",
    "InverseSqrtScheduler",
    "LinearWarmupDecayScheduler",
    "PolynomialDecayScheduler",
    "create_scheduler",
    # Trainer
    "Trainer",
    "TrainerConfig",
    # Distributed
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model_distributed",
    "get_nvlink_info",
    # Hardware
    "HardwareInfo",
    "GPUInfo",
    "detect_hardware",
    "print_hardware_info",
    "setup_h100_optimizations",
    "setup_nccl_optimizations",
    "setup_training_environment",
    "CUDAMemoryManager",
    "is_ampere_or_newer",
    "is_hopper",
    # H100-specific utilities
    "get_optimal_worker_count",
    "get_available_ram_gb",
    "should_preload_dataset",
    "print_h100_optimization_status",
]
