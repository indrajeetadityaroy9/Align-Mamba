"""Fused RMSNorm: single kernel replaces pow(2) -> mean -> rsqrt -> multiply."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. BLOCK_SIZE >= N required."""
    row_idx = tl.program_id(0)

    row_start_x = row_idx * stride_x_row
    row_start_y = row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(X_ptr + row_start_x + col_offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    # FP32 accumulation prevents overflow in variance computation
    x_fp32 = x.to(tl.float32)

    x_sq = x_fp32 * x_fp32
    var = tl.sum(x_sq, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    y = x_fp32 * rstd * w.to(tl.float32)

    tl.store(Y_ptr + row_start_y + col_offsets, y.to(x.dtype), mask=mask)


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight.

    Shape: (..., d_model) -> (..., d_model)
    """
    assert x.is_cuda and weight.is_cuda
    assert x.shape[-1] == weight.shape[0]

    orig_shape = x.shape
    x = x.contiguous()

    hidden_dim = x.shape[-1]
    x_2d = x.view(-1, hidden_dim)
    num_rows = x_2d.shape[0]

    y = torch.empty_like(x_2d)

    # Power-of-2 block size for efficient GPU memory access
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)  # Cap to avoid register spill
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _rmsnorm_fwd_kernel[(num_rows,)](
        x_2d,
        y,
        weight,
        x_2d.stride(0),
        y.stride(0),
        hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y.view(orig_shape)
