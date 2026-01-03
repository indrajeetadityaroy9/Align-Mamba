"""
Training Objectives and Schedulers for Document-Level NMT.

This file contains:
- LabelSmoothingCrossEntropy: Standard NMT loss with smoothing
- SequenceLoss, PackedSequenceLoss: Sequence-level wrappers
- CosineAnnealingWarmupScheduler: Standard scheduler (cosine with warmup)
- InverseSqrtScheduler: Original Transformer scheduler
- create_loss_fn, create_scheduler: Factory functions
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler


# =============================================================================
# Loss Functions
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.

    Standard for NMT training:
    - Prevents model overconfidence
    - Improves generalization
    - Smoothing=0.1 is typical for translation
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy loss.

        Args:
            logits: Model output (batch, seq_len, vocab_size) or (total_tokens, vocab_size)
            targets: Target indices (batch, seq_len) or (total_tokens,)

        Returns:
            Loss tensor
        """
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        vocab_size = logits.size(-1)

        mask = targets != self.ignore_index
        valid_targets = targets.clone()
        valid_targets[~mask] = 0

        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (vocab_size - 1))
            smooth_targets.scatter_(1, valid_targets.unsqueeze(1), self.confidence)

        loss = -torch.sum(log_probs * smooth_targets, dim=-1)
        loss = loss * mask.float()

        if self.reduction == "mean":
            return loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SequenceLoss(nn.Module):
    """Sequence-level loss wrapper."""

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.criterion(logits, labels)


class PackedSequenceLoss(nn.Module):
    """Loss function optimized for packed sequences."""

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.criterion(logits, labels)


def create_loss_fn(
    loss_type: str = "label_smoothing",
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> nn.Module:
    """
    Factory function for loss functions.

    Args:
        loss_type: "label_smoothing", "cross_entropy", "sequence", "packed"
        smoothing: Label smoothing factor
        ignore_index: Index to ignore

    Returns:
        Loss module
    """
    if loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=smoothing, ignore_index=ignore_index)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "sequence":
        return SequenceLoss(smoothing=smoothing, ignore_index=ignore_index)
    elif loss_type == "packed":
        return PackedSequenceLoss(smoothing=smoothing, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

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
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
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
        self.warmup_steps = warmup_steps
        self.scale = scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        warmup_factor = step * (self.warmup_steps ** -1.5)
        decay_factor = step ** -0.5
        factor = self.scale * min(warmup_factor, decay_factor)
        return [base_lr * factor for base_lr in self.base_lrs]


class LinearWarmupDecayScheduler(LRScheduler):
    """Linear warmup followed by linear decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 4000,
        decay_steps: int = 96000,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        elif step < self.warmup_steps + self.decay_steps:
            progress = (step - self.warmup_steps) / self.decay_steps
            scale = 1.0 - progress
        else:
            scale = 0.0

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class PolynomialDecayScheduler(LRScheduler):
    """Polynomial decay with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 4000,
        max_steps: int = 100000,
        min_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
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
