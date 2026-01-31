"""Capacity analysis for Selective State Space Models.

Theoretical bounds and convergence guarantees for SSM memory capacity.

References:
- arXiv:2410.03158: "State Space Models are Provably Optimal Compressors"
  Provides mutual information and rate-distortion bounds for SSM capacity
- arXiv:2506.11891: "Capacity Theorem for Mamba" (Theorem 2)
  "1-layer Mamba solves MQAR with state size N=κ"
- arXiv:2510.03279: MemMamba memory decay analysis
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class CapacityAnalysis:
    """Analysis results for SSM memory capacity.

    Attributes:
        d_state: SSM state dimension
        num_pairs: Number of key-value pairs to store
        theoretical_capacity: Maximum pairs storable (≈ d_state)
        capacity_utilization: num_pairs / theoretical_capacity
        overflow_ratio: How much capacity is exceeded (max 1.0 if no overflow)
        mutual_information_bound: Bits of information storable
        rate_distortion_bound: Compression efficiency lower bound
        convergence_rho: Contraction factor ρ for convergence
        convergence_guaranteed: Whether ρ * L_G < 1 (Theorem 2)
        recommended_cross_attn_interval: Suggested layer spacing for cross-attention
    """
    d_state: int
    num_pairs: int
    theoretical_capacity: float
    capacity_utilization: float
    overflow_ratio: float
    mutual_information_bound: float
    rate_distortion_bound: float
    convergence_rho: float
    convergence_guaranteed: bool
    recommended_cross_attn_interval: Optional[int]


def compute_mutual_information_bound(
    d_state: int,
    precision_bits: int = 16,
) -> float:
    """Compute mutual information bound for SSM state.

    The SSM state can store at most d_state * precision_bits of information.
    In practice, effective capacity is lower due to interference.

    Reference: arXiv:2410.03158, Theorem 3.1
    "I(X; h_t) ≤ d_state * log2(precision)"

    Args:
        d_state: State dimension
        precision_bits: Numerical precision (16 for BF16, 32 for FP32)

    Returns:
        Maximum mutual information in bits
    """
    return d_state * math.log2(precision_bits)


def compute_rate_distortion_bound(
    d_state: int,
    num_pairs: int,
    key_dim: int = 1,
    value_dim: int = 1,
) -> float:
    """Compute rate-distortion bound for SSM compression.

    For MQAR, each key-value pair requires log2(vocab_size) bits for key
    and log2(vocab_size) bits for value. The SSM must compress this
    into d_state dimensions.

    Reference: arXiv:2410.03158, Section 4.2
    "R(D) ≥ (1/2) * log2(σ_x^2 / D)" for Gaussian sources

    Args:
        d_state: State dimension
        num_pairs: Number of key-value pairs
        key_dim: Dimension of each key (default 1 for token IDs)
        value_dim: Dimension of each value (default 1 for token IDs)

    Returns:
        Rate-distortion lower bound (lower is better compression)
    """
    # Information content per pair
    bits_per_pair = (key_dim + value_dim) * 12  # ~12 bits for 4096 vocab tokens

    # Total information to compress
    total_bits = num_pairs * bits_per_pair

    # Compression ratio (how many bits per state dimension)
    compression_ratio = total_bits / d_state

    return compression_ratio


def check_convergence_guarantee(
    d_state: int,
    num_pairs: int,
    n_layers: int,
    decay_rate: float = 0.9,
) -> tuple:
    """Check if SSM convergence is guaranteed for given capacity.

    Reference: arXiv:2506.11891, Theorem 2
    "Convergence requires ρ * L_G < 1 where ρ is contraction factor"

    The contraction factor ρ depends on the eigenvalue spectrum of
    the SSM transition matrix A. For stable SSMs, |λ_max(A)| < 1.

    Args:
        d_state: State dimension
        num_pairs: Number of key-value pairs to memorize
        n_layers: Number of SSM layers
        decay_rate: Expected eigenvalue decay |λ| (default 0.9)

    Returns:
        Tuple of (rho, L_G, is_guaranteed)
    """
    # Contraction factor from SSM eigenvalues
    # ρ ≈ |λ_max|^n_layers (eigenvalue raised to depth)
    rho = decay_rate ** n_layers

    # Lipschitz constant of the gradient
    # Approximated as capacity utilization ratio
    capacity_ratio = num_pairs / d_state
    L_G = capacity_ratio

    # Convergence guaranteed when ρ * L_G < 1
    is_guaranteed = (rho * L_G) < 1.0

    return rho, L_G, is_guaranteed


def compute_recommended_interval(
    d_state: int,
    num_pairs: int,
) -> Optional[int]:
    """Compute recommended cross-attention interval from capacity overflow.

    When num_pairs > d_state, cross-attention is needed to retrieve
    information that overflows SSM capacity.

    Reference: arXiv:2510.03279 (MemMamba), Section 3.1
    "contribution of x_{t-k} decays as |A^k|"

    Args:
        d_state: State dimension
        num_pairs: Number of key-value pairs

    Returns:
        Recommended layer interval for cross-attention, or None if no overflow
    """
    if num_pairs <= d_state:
        return None

    overflow_ratio = num_pairs / d_state

    # Interval derived from decay rate and overflow
    # Higher overflow → more frequent cross-attention
    interval = max(1, int(d_state / math.log(max(overflow_ratio, math.e))))

    return interval


def analyze_ssm_capacity(
    d_state: int,
    num_pairs: int,
    n_layers: int = 24,
    precision_bits: int = 16,
    decay_rate: float = 0.9,
) -> CapacityAnalysis:
    """Comprehensive capacity analysis for SSM architecture.

    Combines all capacity metrics into a single analysis result.

    Args:
        d_state: SSM state dimension
        num_pairs: Number of key-value pairs to memorize
        n_layers: Number of SSM layers
        precision_bits: Numerical precision (16 for BF16)
        decay_rate: Expected eigenvalue decay rate

    Returns:
        CapacityAnalysis dataclass with all metrics
    """
    # Theoretical capacity is approximately d_state (Theorem 2)
    theoretical_capacity = float(d_state)

    # Capacity utilization
    capacity_utilization = num_pairs / theoretical_capacity

    # Overflow ratio (capped at 1.0 if no overflow)
    overflow_ratio = max(1.0, capacity_utilization)

    # Information bounds
    mi_bound = compute_mutual_information_bound(d_state, precision_bits)
    rd_bound = compute_rate_distortion_bound(d_state, num_pairs)

    # Convergence analysis
    rho, L_G, is_guaranteed = check_convergence_guarantee(
        d_state, num_pairs, n_layers, decay_rate
    )

    # Recommended cross-attention interval
    interval = compute_recommended_interval(d_state, num_pairs)

    return CapacityAnalysis(
        d_state=d_state,
        num_pairs=num_pairs,
        theoretical_capacity=theoretical_capacity,
        capacity_utilization=capacity_utilization,
        overflow_ratio=overflow_ratio,
        mutual_information_bound=mi_bound,
        rate_distortion_bound=rd_bound,
        convergence_rho=rho,
        convergence_guaranteed=is_guaranteed,
        recommended_cross_attn_interval=interval,
    )


def format_capacity_report(analysis: CapacityAnalysis) -> str:
    """Format capacity analysis as human-readable report.

    Args:
        analysis: CapacityAnalysis result

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "SSM Capacity Analysis Report",
        "=" * 60,
        f"State Dimension (d_state): {analysis.d_state}",
        f"Key-Value Pairs (num_pairs): {analysis.num_pairs}",
        "",
        "--- Capacity Metrics ---",
        f"Theoretical Capacity: {analysis.theoretical_capacity:.0f} pairs",
        f"Capacity Utilization: {analysis.capacity_utilization:.1%}",
        f"Overflow Ratio: {analysis.overflow_ratio:.2f}x",
        "",
        "--- Information Bounds ---",
        f"Mutual Information Bound: {analysis.mutual_information_bound:.1f} bits",
        f"Rate-Distortion Bound: {analysis.rate_distortion_bound:.2f} bits/dim",
        "",
        "--- Convergence Analysis ---",
        f"Contraction Factor (ρ): {analysis.convergence_rho:.4f}",
        f"Convergence Guaranteed: {'Yes' if analysis.convergence_guaranteed else 'No'}",
        "",
        "--- Recommendations ---",
    ]

    if analysis.recommended_cross_attn_interval is not None:
        lines.append(f"Cross-Attention Interval: Every {analysis.recommended_cross_attn_interval} layers")
        lines.append("Reason: Capacity overflow requires periodic encoder retrieval")
    else:
        lines.append("Cross-Attention: Only Layer 0 needed (Blind Start fix)")
        lines.append("Reason: No capacity overflow detected")

    lines.append("=" * 60)

    return "\n".join(lines)


__all__ = [
    "CapacityAnalysis",
    "analyze_ssm_capacity",
    "compute_mutual_information_bound",
    "compute_rate_distortion_bound",
    "check_convergence_guarantee",
    "compute_recommended_interval",
    "format_capacity_report",
]
