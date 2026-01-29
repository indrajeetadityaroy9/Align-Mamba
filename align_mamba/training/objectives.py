"""Training objectives and schedulers for State Capacity experiments."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler

from align_mamba.kernels import fused_cross_entropy_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy with fused Triton kernel on GPU."""

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
        if logits.is_cuda:
            return fused_cross_entropy_loss(
                logits,
                targets,
                smoothing=self.smoothing,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )

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
        return loss


class CosineAnnealingWarmupScheduler(LRScheduler):
    """Cosine annealing with linear warmup."""

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

        return [self.min_lr + (base_lr - self.min_lr) * scale for base_lr in self.base_lrs]


def create_loss_fn(smoothing: float = 0.1, ignore_index: int = -100) -> nn.Module:
    """Create label smoothing cross entropy loss."""
    return LabelSmoothingCrossEntropy(smoothing=smoothing, ignore_index=ignore_index)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 4000,
    max_steps: int = 100000,
    min_lr: float = 1e-6,
) -> LRScheduler:
    """Create cosine annealing scheduler with warmup."""
    return CosineAnnealingWarmupScheduler(
        optimizer, warmup_steps=warmup_steps, max_steps=max_steps, min_lr=min_lr
    )
