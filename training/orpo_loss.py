"""
ORPO (Odds Ratio Preference Optimization) Loss for NMT.

ORPO combines supervised fine-tuning with preference optimization:
L_ORPO = L_NLL(chosen) + β * L_OR(chosen, rejected)

Reference: https://arxiv.org/abs/2403.07691
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor,
                           pad_idx: int) -> torch.Tensor:
    """
    Compute average log probability per sequence (excluding padding).

    Args:
        logits: (batch, seq_len, vocab_size) model output logits
        labels: (batch, seq_len) target token indices
        pad_idx: padding token index to ignore

    Returns:
        seq_log_probs: (batch,) average log prob per sequence
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for actual tokens
    # labels: (batch, seq_len) -> (batch, seq_len, 1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Create mask for non-padding tokens
    mask = (labels != pad_idx).float()

    # Sum log probs and normalize by sequence length
    seq_log_probs = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return seq_log_probs


class ORPOLoss(nn.Module):
    """
    ORPO Loss: NLL on chosen + odds ratio preference loss.

    Args:
        beta: Weight for the odds ratio term (default: 0.1)
        pad_idx: Padding token index for masking
    """

    def __init__(self, beta: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.beta = beta
        self.pad_idx = pad_idx

    def forward(
        self,
        chosen_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_logits: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute ORPO loss.

        Args:
            chosen_logits: (batch, seq_len, vocab_size) logits for chosen sequences
            chosen_labels: (batch, seq_len) target tokens for chosen
            rejected_logits: (batch, seq_len, vocab_size) logits for rejected sequences
            rejected_labels: (batch, seq_len) target tokens for rejected

        Returns:
            loss: Combined ORPO loss
            metrics: Dict with individual loss components for logging
        """
        # 1. NLL loss on chosen (supervised language modeling)
        # Reshape for cross entropy: (batch * seq_len, vocab_size)
        batch_size, seq_len, vocab_size = chosen_logits.shape
        nll_loss = F.cross_entropy(
            chosen_logits.reshape(-1, vocab_size),
            chosen_labels.reshape(-1),
            ignore_index=self.pad_idx,
            reduction='mean'
        )

        # 2. Compute sequence-level log probabilities
        chosen_log_probs = get_sequence_log_probs(
            chosen_logits, chosen_labels, self.pad_idx
        )
        rejected_log_probs = get_sequence_log_probs(
            rejected_logits, rejected_labels, self.pad_idx
        )

        # 3. Odds ratio loss
        # log(odds_chosen / odds_rejected) = log_prob_chosen - log_prob_rejected
        log_odds_ratio = chosen_log_probs - rejected_log_probs

        # ORPO uses log-sigmoid of the odds ratio
        # L_OR = -log(sigmoid(log_odds_ratio)) = -logsigmoid(log_odds_ratio)
        orpo_loss = -F.logsigmoid(log_odds_ratio).mean()

        # 4. Combined loss
        total_loss = nll_loss + self.beta * orpo_loss

        # Metrics for monitoring
        metrics = {
            'nll': nll_loss.item(),
            'orpo': orpo_loss.item(),
            'log_odds': log_odds_ratio.mean().item(),
            'chosen_log_prob': chosen_log_probs.mean().item(),
            'rejected_log_prob': rejected_log_probs.mean().item(),
        }

        return total_loss, metrics
