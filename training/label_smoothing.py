"""Label Smoothing Loss for NMT.

Label smoothing prevents overconfident predictions by distributing
a small probability mass to non-target tokens, improving generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss.

    Instead of one-hot targets, uses:
        target_smooth = (1 - epsilon) * one_hot + epsilon / vocab_size

    This prevents the model from becoming overconfident and improves
    calibration, especially important for NMT where uncertainty matters.

    Args:
        vocab_size: Size of vocabulary
        padding_idx: Index of padding token (ignored in loss)
        smoothing: Smoothing parameter epsilon (typically 0.1)
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            logits: (batch, vocab_size) or (batch, seq_len, vocab_size) model outputs
            target: (batch,) or (batch, seq_len) target token indices

        Returns:
            Scalar loss tensor
        """
        # Handle different input shapes
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            target = target.view(-1)

        # Log softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smooth target distribution
        # Uniform distribution: epsilon / vocab_size for all tokens
        smooth_target = torch.full_like(log_probs, self.smoothing / self.vocab_size)

        # Add confidence to true target
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)

        # Zero out padding positions in target distribution
        pad_mask = target.eq(self.padding_idx)
        smooth_target.masked_fill_(pad_mask.unsqueeze(1), 0.0)

        # KL divergence: sum over vocab, mean over non-padding tokens
        loss = -torch.sum(smooth_target * log_probs, dim=-1)

        # Average over non-padding tokens
        non_pad_mask = ~pad_mask
        loss = loss.masked_select(non_pad_mask).mean()

        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Alternative implementation using KL divergence formulation.

    More efficient for large vocabularies as it doesn't create
    full smooth target distribution.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        padding_idx: int = 0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smoothing = smoothing
        self.padding_idx = padding_idx
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Efficient label smoothing using decomposition.

        L_smooth = (1-ε) * CE(p, y) + ε * H(p)

        where H(p) is the entropy term (sum of all log probs).
        """
        # Handle different input shapes
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            target = target.view(-1)

        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Standard cross-entropy term
        nll_loss = F.nll_loss(
            log_probs, target,
            ignore_index=self.padding_idx,
            reduction='none'
        )

        # Smoothing term: negative mean of all log probs (entropy-like)
        smooth_loss = -log_probs.mean(dim=-1)

        # Combine: (1-ε) * CE + ε * smooth_term
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        # Handle padding
        pad_mask = target.eq(self.padding_idx)
        loss = loss.masked_fill(pad_mask, 0.0)

        if self.reduction == 'mean':
            return loss.sum() / (~pad_mask).sum().float().clamp(min=1.0)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
