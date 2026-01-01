"""
Learning Rate Schedulers for Document-Level NMT Training.

Cosine annealing with warmup is standard for Transformer/Mamba training.
"""

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmupScheduler(LRScheduler):
    """
    Cosine annealing with linear warmup.

    Standard scheduler for transformer-based models:
    - Linear warmup for first `warmup_steps`
    - Cosine decay to `min_lr` after warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 4000,
        max_steps: int = 100000,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Total training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch for resume
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class InverseSqrtScheduler(LRScheduler):
    """
    Inverse square root scheduler (Transformer original).

    lr = base_lr * min(step^-0.5, step * warmup^-1.5)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 4000,
        scale: float = 1.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            scale: Learning rate scale
            last_epoch: Last epoch for resume
        """
        self.warmup_steps = warmup_steps
        self.scale = scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        step = max(1, self.last_epoch + 1)
        warmup_factor = step * (self.warmup_steps ** -1.5)
        decay_factor = step ** -0.5
        factor = self.scale * min(warmup_factor, decay_factor)

        return [base_lr * factor for base_lr in self.base_lrs]


class LinearWarmupDecayScheduler(LRScheduler):
    """
    Linear warmup followed by linear decay.

    Simple alternative to cosine scheduling.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 4000,
        decay_steps: int = 96000,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            decay_steps: Steps to decay (total = warmup + decay)
            min_lr: Minimum learning rate
            last_epoch: Last epoch for resume
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        elif step < self.warmup_steps + self.decay_steps:
            # Linear decay
            progress = (step - self.warmup_steps) / self.decay_steps
            scale = 1.0 - progress
        else:
            scale = 0.0

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class PolynomialDecayScheduler(LRScheduler):
    """
    Polynomial decay with warmup.

    More aggressive early decay than cosine.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 4000,
        max_steps: int = 100000,
        min_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Total training steps
            min_lr: Minimum learning rate
            power: Polynomial power (1.0 = linear, 2.0 = quadratic)
            last_epoch: Last epoch for resume
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Polynomial decay
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = (1.0 - min(1.0, progress)) ** self.power

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


def create_scheduler(
    scheduler_type: str = "cosine",
    optimizer: torch.optim.Optimizer = None,
    warmup_steps: int = 4000,
    max_steps: int = 100000,
    min_lr: float = 1e-6,
    **kwargs,
) -> LRScheduler:
    """
    Factory function for schedulers.

    Args:
        scheduler_type: "cosine", "inverse_sqrt", "linear", "polynomial"
        optimizer: PyTorch optimizer
        warmup_steps: Warmup steps
        max_steps: Total training steps
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler arguments

    Returns:
        LR Scheduler
    """
    if scheduler_type == "cosine":
        return CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "inverse_sqrt":
        return InverseSqrtScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            **kwargs,
        )
    elif scheduler_type == "linear":
        return LinearWarmupDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            decay_steps=max_steps - warmup_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "polynomial":
        return PolynomialDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_lr,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
