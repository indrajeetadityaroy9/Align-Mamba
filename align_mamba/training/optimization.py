"""Training optimization: objectives, schedulers, and adaptive hyperparameters."""

import math
from typing import Dict, Iterator

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from align_mamba.kernels.loss import fused_cross_entropy_loss


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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        assert logits.is_cuda, "LabelSmoothingCrossEntropy requires CUDA tensors"
        return fused_cross_entropy_loss(
            logits,
            targets,
            smoothing=self.smoothing,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


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


def compute_adaptive_dropout(num_params: int, num_samples: int) -> float:
    """Dropout from capacity/data ratio. Reference: Srivastava et al., 2014."""
    capacity_ratio = num_params / max(num_samples, 1)
    dropout = 0.5 * (1 - math.exp(-capacity_ratio * 100))
    return max(0.0, min(0.5, dropout))


def compute_per_param_weight_decay(param: torch.Tensor, base_decay: float = 0.01) -> float:
    """Scale weight decay by magnitude. Reference: arXiv 1711.05101."""
    param_norm = param.norm().item()
    return base_decay / max(param_norm, 1e-4)


def create_adaptive_param_groups(
    model: nn.Module,
    base_lr: float,
    base_decay: float = 0.01,
) -> list:
    """Create optimizer param groups with per-parameter weight decay."""
    no_decay_keywords = ["bias", "LayerNorm", "RMSNorm", "norm"]

    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        skip_decay = any(kw in name for kw in no_decay_keywords)

        if skip_decay:
            param_groups.append({
                "params": [param],
                "lr": base_lr,
                "weight_decay": 0.0,
            })
        else:
            decay = compute_per_param_weight_decay(param, base_decay)
            param_groups.append({
                "params": [param],
                "lr": base_lr,
                "weight_decay": decay,
            })

    return param_groups


def compute_label_smoothing_from_entropy(vocab_size: int, target_entropy_increase: float = 0.05) -> float:
    """Entropy-based smoothing. Reference: arXiv 1906.02629."""
    H_max = math.log(vocab_size)
    smoothing = target_entropy_increase * H_max / (H_max + 1)
    return max(0.01, min(0.2, smoothing))


def compute_agc_factor(param: torch.Tensor) -> float:
    """AGC clip factor from parameter std. Reference: arXiv 2102.06171."""
    param_std = param.std().item()
    fan_in = param.shape[-1] if param.dim() > 1 else 1
    return param_std / (fan_in ** 0.5 + 1e-4)


def adaptive_gradient_clip_(parameters: Iterator[nn.Parameter], eps: float = 1e-3) -> None:
    """Per-parameter AGC. Reference: arXiv 2102.06171."""
    for p in parameters:
        if p.grad is None:
            continue

        param_norm = p.data.norm(p=2).clamp(min=eps)
        grad_norm = p.grad.data.norm(p=2)

        clip_factor = compute_agc_factor(p.data)
        max_norm = param_norm * clip_factor

        if grad_norm > max_norm:
            p.grad.data.mul_(max_norm / (grad_norm + eps))


def compute_logging_intervals(
    num_samples: int,
    batch_size: int,
    target_logs_per_epoch: int = 100,
    target_evals_per_epoch: int = 10,
    target_saves_per_epoch: int = 2,
) -> Dict[str, int]:
    """Derive logging intervals from dataset size."""
    steps_per_epoch = max(1, num_samples // batch_size)

    return {
        "log_steps": max(1, steps_per_epoch // target_logs_per_epoch),
        "eval_steps": max(10, steps_per_epoch // target_evals_per_epoch),
        "save_steps": max(100, steps_per_epoch // target_saves_per_epoch),
    }
