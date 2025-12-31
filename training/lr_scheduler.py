"""Learning Rate Schedulers for NMT Training.

Implements warmup + cosine decay scheduling, which is standard
for Transformer training and provides stable convergence.
"""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.

    Schedule:
    - Warmup phase: LR increases linearly from 0 to peak_lr
    - Decay phase: LR decreases following cosine curve to min_lr

    This is the standard schedule for Transformer models and
    provides stable training with good final convergence.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        peak_lr: Maximum learning rate (reached at end of warmup)
        min_lr: Minimum learning rate (reached at end of training)
        last_epoch: Last epoch index (for resuming)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float = 0.0003,
        min_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        # Store base lrs before calling super().__init__
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [self.peak_lr * warmup_factor for _ in self.base_lrs]
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            current_step = step - self.warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / decay_steps))
            decayed_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
            return [decayed_lr for _ in self.base_lrs]


class WarmupInverseSquareRootScheduler(LRScheduler):
    """
    Learning rate scheduler with warmup and inverse square root decay.

    This is the original Transformer schedule from "Attention Is All You Need":
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Simplified to:
    - Warmup: lr increases linearly
    - Decay: lr decreases as 1/sqrt(step)

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        peak_lr: Maximum learning rate
        last_epoch: Last epoch index
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        peak_lr: float = 0.0003,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)

        if step < self.warmup_steps:
            # Linear warmup
            return [self.peak_lr * step / self.warmup_steps for _ in self.base_lrs]
        else:
            # Inverse square root decay
            decay_factor = math.sqrt(self.warmup_steps / step)
            return [self.peak_lr * decay_factor for _ in self.base_lrs]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 1.5,
    peak_lr: float = 0.0003,
    min_lr: float = 1e-7
) -> LRScheduler:
    """
    Factory function to create LR scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'inverse_sqrt', 'step')
        num_epochs: Total number of training epochs
        steps_per_epoch: Number of optimization steps per epoch
        warmup_epochs: Number of warmup epochs (can be fractional)
        peak_lr: Peak learning rate
        min_lr: Minimum learning rate (for cosine)

    Returns:
        Configured LR scheduler
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    if scheduler_type == 'cosine':
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=peak_lr,
            min_lr=min_lr
        )
    elif scheduler_type == 'inverse_sqrt':
        return WarmupInverseSquareRootScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            peak_lr=peak_lr
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=3 * steps_per_epoch, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class SchedulerWithWarmupWrapper:
    """
    Wrapper that adds warmup to any existing scheduler.

    Useful for adding warmup to schedulers that don't support it natively.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: LRScheduler,
        warmup_steps: int,
        warmup_start_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0

        # Store target LRs from base scheduler
        self.target_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            warmup_factor = self.current_step / self.warmup_steps
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.warmup_start_lr + (self.target_lrs[i] - self.warmup_start_lr) * warmup_factor
        else:
            # Use base scheduler
            self.base_scheduler.step()

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'base_scheduler': self.base_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
