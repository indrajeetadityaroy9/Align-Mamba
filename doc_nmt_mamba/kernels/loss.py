"""Fused cross-entropy with label smoothing via online softmax."""

import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_fwd_kernel(
    LOGITS_ptr,
    LABELS_ptr,
    LOSSES_ptr,
    stride_logits_batch,
    stride_logits_vocab,
    vocab_size,
    smoothing,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    """
    One program per sample. Online log-sum-exp avoids materializing
    full softmax distribution (critical for 32K+ vocab).
    """
    batch_idx = tl.program_id(0)

    label = tl.load(LABELS_ptr + batch_idx)
    is_ignored = label == ignore_index

    # Online softmax: track running max (m) and scaled sum (d)
    # Final log-sum-exp = m + log(d)
    m = float("-inf")
    d = 0.0

    logits_row_ptr = LOGITS_ptr + batch_idx * stride_logits_batch
    target_logit = float("-inf")

    for v_start in range(0, vocab_size, BLOCK_V):
        v_idx = v_start + tl.arange(0, BLOCK_V)
        mask = v_idx < vocab_size

        logits_block = tl.load(
            logits_row_ptr + v_idx * stride_logits_vocab,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

        is_target = (v_idx == label) & mask
        target_logit = tl.where(
            tl.max(is_target.to(tl.int32), axis=0) > 0,
            tl.sum(tl.where(is_target, logits_block, 0.0), axis=0),
            target_logit,
        )

        # Numerically stable online update: rescale old sum when max increases
        block_max = tl.max(logits_block, axis=0)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(logits_block - m_new), axis=0)
        m = m_new

    log_sum_exp = m + tl.log(d)
    target_log_prob = target_logit - log_sum_exp

    hard_loss = -target_log_prob
    # Smooth component: KL from uniform ~ log(vocab_size)
    log_vocab = tl.log(vocab_size.to(tl.float32))
    smooth_loss = log_vocab

    loss = (1.0 - smoothing) * hard_loss + smoothing * smooth_loss
    loss = tl.where(is_ignored, 0.0, loss)

    tl.store(LOSSES_ptr + batch_idx, loss)


def fused_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.1,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross-entropy with label smoothing using online softmax.

    Shapes: logits (B, V) or (B, S, V), labels (B,) or (B, S)
    Returns: scalar (mean/sum) or (B*S,) tensor (none)
    """
    assert logits.is_cuda and labels.is_cuda

    if logits.dim() == 3:
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
    else:
        vocab_size = logits.shape[-1]

    logits = logits.contiguous()
    labels = labels.contiguous()

    batch_size = logits.shape[0]
    losses = torch.empty(batch_size, device=logits.device, dtype=torch.float32)

    # Larger blocks = fewer iterations but more register pressure
    BLOCK_V = min(triton.next_power_of_2(vocab_size), 4096)

    _cross_entropy_fwd_kernel[(batch_size,)](
        logits,
        labels,
        losses,
        logits.stride(0),
        logits.stride(1),
        vocab_size,
        smoothing,
        ignore_index,
        BLOCK_V=BLOCK_V,
    )

    if reduction == "mean":
        valid_mask = labels != ignore_index
        num_valid = valid_mask.sum().clamp(min=1)
        return losses.sum() / num_valid
    elif reduction == "sum":
        return losses.sum()
    else:
        return losses
