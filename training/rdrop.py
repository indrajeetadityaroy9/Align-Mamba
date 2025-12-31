"""R-Drop: Regularized Dropout for NMT.

R-Drop (https://arxiv.org/abs/2106.14448) is a simple but effective
regularization technique that enforces consistency between two
forward passes with different dropout masks.

Key insight: The model should produce similar outputs regardless
of which neurons are dropped out, making it more robust.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    padding_mask: torch.Tensor = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute symmetric KL divergence between two distributions.

    KL_sym = 0.5 * (KL(P||Q) + KL(Q||P))

    Args:
        p_logits: (batch, seq_len, vocab_size) first forward pass logits
        q_logits: (batch, seq_len, vocab_size) second forward pass logits
        padding_mask: (batch, seq_len) True for padding positions
        temperature: Temperature for softmax (default 1.0)

    Returns:
        Scalar KL divergence loss
    """
    # Apply temperature
    p_logits = p_logits / temperature
    q_logits = q_logits / temperature

    # Compute log probabilities
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    q_log_probs = F.log_softmax(q_logits, dim=-1)

    # Compute probabilities
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    # KL(P||Q) = sum(P * (log P - log Q))
    kl_pq = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)
    # KL(Q||P)
    kl_qp = torch.sum(q_probs * (q_log_probs - p_log_probs), dim=-1)

    # Symmetric KL
    kl_loss = 0.5 * (kl_pq + kl_qp)

    # Mask padding
    if padding_mask is not None:
        kl_loss = kl_loss.masked_fill(padding_mask, 0.0)
        num_tokens = (~padding_mask).sum().float().clamp(min=1.0)
        return kl_loss.sum() / num_tokens

    return kl_loss.mean()


class RDropLoss(nn.Module):
    """
    R-Drop loss combining cross-entropy with KL divergence regularization.

    Total loss = CE_loss + alpha * KL_loss

    where KL_loss enforces consistency between two forward passes
    with different dropout masks.

    Usage:
        rdrop_loss = RDropLoss(alpha=0.7)

        # Two forward passes (dropout creates different masks)
        logits1 = model(input)
        logits2 = model(input)

        loss = rdrop_loss(logits1, logits2, targets, padding_mask)

    Args:
        alpha: Weight for KL divergence term (0.5-1.0 typical)
        ce_loss_fn: Cross-entropy loss function (optional)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        ce_loss_fn: nn.Module = None,
        kl_temperature: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.ce_loss_fn = ce_loss_fn
        self.kl_temperature = kl_temperature

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> tuple:
        """
        Compute R-Drop loss.

        Args:
            logits1: (batch, seq_len, vocab_size) first forward pass
            logits2: (batch, seq_len, vocab_size) second forward pass
            targets: (batch, seq_len) target token indices
            padding_mask: (batch, seq_len) True for padding positions

        Returns:
            total_loss: Combined CE + KL loss
            ce_loss: Cross-entropy component
            kl_loss: KL divergence component
        """
        # Compute CE loss for both passes
        if self.ce_loss_fn is not None:
            ce_loss1 = self.ce_loss_fn(logits1.view(-1, logits1.size(-1)), targets.view(-1))
            ce_loss2 = self.ce_loss_fn(logits2.view(-1, logits2.size(-1)), targets.view(-1))
            ce_loss = 0.5 * (ce_loss1 + ce_loss2)
        else:
            # Use default CE with ignore_index for padding
            ce_loss1 = F.cross_entropy(
                logits1.view(-1, logits1.size(-1)),
                targets.view(-1),
                reduction='mean'
            )
            ce_loss2 = F.cross_entropy(
                logits2.view(-1, logits2.size(-1)),
                targets.view(-1),
                reduction='mean'
            )
            ce_loss = 0.5 * (ce_loss1 + ce_loss2)

        # Compute KL divergence
        kl_loss = compute_kl_divergence(
            logits1, logits2,
            padding_mask=padding_mask,
            temperature=self.kl_temperature
        )

        # Combined loss
        total_loss = ce_loss + self.alpha * kl_loss

        return total_loss, ce_loss, kl_loss


def rdrop_forward_pass(
    encoder,
    decoder,
    src_batch: torch.Tensor,
    tgt_batch: torch.Tensor,
    src_lengths: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Single forward pass through encoder-decoder with teacher forcing.

    This is a helper for R-Drop training where we need two identical
    forward passes (but with different dropout masks).

    Args:
        encoder: Encoder model
        decoder: Decoder model
        src_batch: (batch, src_len) source tokens
        tgt_batch: (batch, tgt_len) target tokens
        src_lengths: (batch,) source lengths
        device: Torch device

    Returns:
        logits: (batch, tgt_len-1, vocab_size) output logits
    """
    encoder_outputs, encoder_hidden = encoder(src_batch, src_lengths)

    batch_size = src_batch.size(0)
    tgt_len = tgt_batch.size(1)

    # Initialize decoder
    decoder_input = tgt_batch[:, 0]
    decoder_hidden = encoder_hidden

    all_logits = []

    # Teacher forcing
    for t in range(1, tgt_len):
        output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        all_logits.append(output.squeeze(1))
        decoder_input = tgt_batch[:, t]

    # Stack all outputs
    logits = torch.stack(all_logits, dim=1)
    # logits: (batch, tgt_len-1, vocab_size)

    return logits
