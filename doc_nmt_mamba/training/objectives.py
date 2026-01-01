"""
Training objectives for Document-Level NMT.

Label smoothing helps prevent overconfidence and improves generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
        """
        Args:
            smoothing: Label smoothing factor (0.1 typical for NMT)
            ignore_index: Index to ignore in loss computation
            reduction: Reduction mode ("mean", "sum", "none")
        """
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
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        vocab_size = logits.size(-1)

        # Create mask for valid positions
        mask = targets != self.ignore_index
        valid_targets = targets.clone()
        valid_targets[~mask] = 0  # Avoid indexing errors

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot with smoothing
        # smooth_targets[i] = smoothing / (vocab_size - 1) for all classes except target
        # smooth_targets[target] = confidence + smoothing / (vocab_size - 1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (vocab_size - 1))
            smooth_targets.scatter_(1, valid_targets.unsqueeze(1), self.confidence)

        # Compute loss
        loss = -torch.sum(log_probs * smooth_targets, dim=-1)

        # Apply mask
        loss = loss * mask.float()

        if self.reduction == "mean":
            return loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SequenceLoss(nn.Module):
    """
    Sequence-level loss wrapper.

    Handles both packed and padded sequences uniformly.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        """
        Args:
            smoothing: Label smoothing factor
            ignore_index: Index to ignore
        """
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
        """
        Compute sequence loss.

        Args:
            logits: Model predictions
            labels: Target labels
            mask: Optional mask (for packed sequences, use ignore_index instead)

        Returns:
            Scalar loss
        """
        return self.criterion(logits, labels)


class PackedSequenceLoss(nn.Module):
    """
    Loss function optimized for packed sequences.

    Works directly with cu_seqlens format.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        """
        Args:
            smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss
        """
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
        """
        Compute loss for packed sequences.

        Args:
            logits: Packed predictions (total_tokens, vocab_size)
            labels: Packed labels (total_tokens,)
            cu_seqlens: Cumulative sequence lengths (optional, for logging)

        Returns:
            Scalar loss
        """
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
        return LabelSmoothingCrossEntropy(
            smoothing=smoothing,
            ignore_index=ignore_index,
        )
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "sequence":
        return SequenceLoss(smoothing=smoothing, ignore_index=ignore_index)
    elif loss_type == "packed":
        return PackedSequenceLoss(smoothing=smoothing, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
