"""Unified metrics for training and evaluation."""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class BatchMetrics:
    """Metrics computed for a single batch."""

    token_correct: int
    token_total: int
    sample_correct: int
    sample_total: int

    @property
    def token_accuracy(self) -> float:
        """Token-level accuracy."""
        return self.token_correct / max(self.token_total, 1)

    @property
    def sample_accuracy(self) -> float:
        """Sample-level accuracy (all tokens correct)."""
        return self.sample_correct / max(self.sample_total, 1)

    def __add__(self, other: "BatchMetrics") -> "BatchMetrics":
        """Accumulate metrics across batches."""
        return BatchMetrics(
            token_correct=self.token_correct + other.token_correct,
            token_total=self.token_total + other.token_total,
            sample_correct=self.sample_correct + other.sample_correct,
            sample_total=self.sample_total + other.sample_total,
        )


def compute_batch_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> BatchMetrics:
    """Compute token and sample accuracy for a batch.

    Args:
        predictions: Predicted token IDs (batch, seq_len)
        labels: Ground truth token IDs (batch, seq_len)
        mask: Boolean mask for valid positions (batch, seq_len)

    Returns:
        BatchMetrics with token and sample accuracy
    """
    token_correct = ((predictions == labels) & mask).sum().item()
    token_total = mask.sum().item()
    sample_correct = ((predictions == labels) | ~mask).all(dim=-1).sum().item()
    sample_total = predictions.size(0)

    return BatchMetrics(
        token_correct=int(token_correct),
        token_total=int(token_total),
        sample_correct=int(sample_correct),
        sample_total=int(sample_total),
    )


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity (exp of cross-entropy loss).

    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        labels: Ground truth labels (batch, seq_len)
        ignore_index: Label index to ignore in loss computation

    Returns:
        Perplexity value
    """
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='mean')
    return torch.exp(loss).item()
