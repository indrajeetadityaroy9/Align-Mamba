#!/usr/bin/env python3
"""
Hardware Benchmark Script for Document-Level NMT.

Tests and benchmarks:
- GPU compute capabilities (matmul, attention, Mamba)
- Memory bandwidth and utilization
- Multi-GPU NVLink performance
- torch.compile speedups
- BF16 vs FP32 performance

Usage:
    python scripts/benchmark_hardware.py
    python scripts/benchmark_hardware.py --full  # Run all benchmarks
    python scripts/benchmark_hardware.py --quick  # Quick sanity check
"""

import argparse
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def benchmark_matmul(
    sizes: list = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)],
    dtype: torch.dtype = torch.bfloat16,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> dict:
    """Benchmark matrix multiplication performance."""
    print_header("MATMUL BENCHMARK")
    print(f"Dtype: {dtype}, Warmup: {num_warmup}, Iterations: {num_iters}")

    results = {}

    for m, n in sizes:
        k = n  # Square-ish matmul

        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)

        # Warmup
        for _ in range(num_warmup):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Calculate TFLOPS
        flops = 2 * m * n * k * num_iters
        tflops = flops / elapsed / 1e12

        results[f"{m}x{k}x{n}"] = {
            "time_ms": elapsed * 1000 / num_iters,
            "tflops": tflops,
        }

        print(f"  [{m}x{k}x{n}] {elapsed*1000/num_iters:.3f}ms | {tflops:.1f} TFLOPS")

    return results


def benchmark_attention(
    batch_sizes: list = [1, 4, 8],
    seq_lengths: list = [512, 1024, 2048, 4096],
    d_model: int = 768,
    n_heads: int = 12,
    dtype: torch.dtype = torch.bfloat16,
    num_warmup: int = 5,
    num_iters: int = 20,
) -> dict:
    """Benchmark attention (Flash SDP) performance."""
    print_header("ATTENTION BENCHMARK (Flash SDP)")
    print(f"d_model: {d_model}, n_heads: {n_heads}, dtype: {dtype}")

    results = {}
    head_dim = d_model // n_heads

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            try:
                q = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
                k = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)
                v = torch.randn(batch_size, n_heads, seq_len, head_dim, device="cuda", dtype=dtype)

                # Warmup
                for _ in range(num_warmup):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

                # Benchmark
                start = time.perf_counter()
                for _ in range(num_iters):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                time_ms = elapsed * 1000 / num_iters
                tokens_per_sec = batch_size * seq_len / (elapsed / num_iters)

                results[f"B{batch_size}_L{seq_len}"] = {
                    "time_ms": time_ms,
                    "tokens_per_sec": tokens_per_sec,
                }

                print(f"  [B={batch_size}, L={seq_len}] {time_ms:.3f}ms | {tokens_per_sec/1000:.1f}K tok/s")

            except torch.cuda.OutOfMemoryError:
                print(f"  [B={batch_size}, L={seq_len}] OOM")
                torch.cuda.empty_cache()

    return results


def benchmark_mamba(
    batch_sizes: list = [1, 4, 8],
    seq_lengths: list = [512, 1024, 2048, 4096, 8192],
    d_model: int = 768,
    d_state: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    num_warmup: int = 5,
    num_iters: int = 20,
) -> dict:
    """Benchmark Mamba-2 performance."""
    print_header("MAMBA-2 BENCHMARK")

    try:
        from mamba_ssm import Mamba2
    except ImportError:
        print("  ERROR: mamba-ssm not installed. Skipping.")
        return {}

    print(f"d_model: {d_model}, d_state: {d_state}, dtype: {dtype}")

    results = {}

    # Create Mamba model
    mamba = Mamba2(d_model=d_model, d_state=d_state, device="cuda", dtype=dtype)
    mamba.eval()

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            try:
                x = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype)

                # Warmup
                with torch.no_grad():
                    for _ in range(num_warmup):
                        _ = mamba(x)
                torch.cuda.synchronize()

                # Benchmark
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(num_iters):
                        _ = mamba(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                time_ms = elapsed * 1000 / num_iters
                tokens_per_sec = batch_size * seq_len / (elapsed / num_iters)

                results[f"B{batch_size}_L{seq_len}"] = {
                    "time_ms": time_ms,
                    "tokens_per_sec": tokens_per_sec,
                }

                print(f"  [B={batch_size}, L={seq_len}] {time_ms:.3f}ms | {tokens_per_sec/1000:.1f}K tok/s")

            except torch.cuda.OutOfMemoryError:
                print(f"  [B={batch_size}, L={seq_len}] OOM")
                torch.cuda.empty_cache()

    return results


def benchmark_memory_bandwidth(
    sizes_gb: list = [0.1, 0.5, 1.0, 2.0, 4.0],
    dtype: torch.dtype = torch.bfloat16,
    num_iters: int = 50,
) -> dict:
    """Benchmark GPU memory bandwidth."""
    print_header("MEMORY BANDWIDTH BENCHMARK")
    print(f"Dtype: {dtype}")

    results = {}
    bytes_per_elem = 2 if dtype == torch.bfloat16 else 4

    for size_gb in sizes_gb:
        try:
            numel = int(size_gb * 1e9 / bytes_per_elem)
            a = torch.randn(numel, device="cuda", dtype=dtype)
            b = torch.empty_like(a)

            # Warmup
            for _ in range(5):
                b.copy_(a)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iters):
                b.copy_(a)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            # Calculate bandwidth (read + write)
            bytes_transferred = 2 * numel * bytes_per_elem * num_iters
            bandwidth_gbps = bytes_transferred / elapsed / 1e9

            results[f"{size_gb}GB"] = {
                "bandwidth_gbps": bandwidth_gbps,
            }

            print(f"  [{size_gb}GB tensor] {bandwidth_gbps:.1f} GB/s")

            del a, b
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  [{size_gb}GB tensor] OOM")
            torch.cuda.empty_cache()

    return results


def benchmark_nvlink(
    sizes_mb: list = [1, 10, 100, 1000],
    num_iters: int = 100,
) -> dict:
    """Benchmark NVLink GPU-to-GPU transfer."""
    print_header("NVLINK BENCHMARK")

    if torch.cuda.device_count() < 2:
        print("  Skipping: Need 2+ GPUs for NVLink benchmark")
        return {}

    # Check P2P access
    can_p2p = torch.cuda.can_device_access_peer(0, 1)
    print(f"P2P Access: {'Enabled' if can_p2p else 'Disabled'}")

    if can_p2p:
        # Enable P2P
        torch.cuda.set_device(0)
        torch.cuda.enable_peer_access(1)

    results = {}

    for size_mb in sizes_mb:
        try:
            numel = int(size_mb * 1e6 / 2)  # BF16 = 2 bytes
            src = torch.randn(numel, device="cuda:0", dtype=torch.bfloat16)
            dst = torch.empty(numel, device="cuda:1", dtype=torch.bfloat16)

            # Warmup
            for _ in range(10):
                dst.copy_(src)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iters):
                dst.copy_(src)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bytes_transferred = numel * 2 * num_iters
            bandwidth_gbps = bytes_transferred / elapsed / 1e9

            results[f"{size_mb}MB"] = {
                "bandwidth_gbps": bandwidth_gbps,
            }

            print(f"  [{size_mb}MB] GPU0 -> GPU1: {bandwidth_gbps:.1f} GB/s")

            del src, dst
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{size_mb}MB] Error: {e}")

    return results


def benchmark_compile(
    batch_size: int = 4,
    seq_len: int = 1024,
    d_model: int = 768,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> dict:
    """Benchmark torch.compile speedup."""
    print_header("TORCH.COMPILE BENCHMARK")

    class SimpleMLP(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.fc1 = nn.Linear(d_model, d_model * 4)
            self.fc2 = nn.Linear(d_model * 4, d_model)
            self.act = nn.GELU()

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    results = {}

    # Create model
    model = SimpleMLP(d_model).cuda().bfloat16()
    x = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.bfloat16)

    # Benchmark eager mode
    model.eval()
    for _ in range(num_warmup):
        _ = model(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = model(x)
    torch.cuda.synchronize()
    eager_time = (time.perf_counter() - start) * 1000 / num_iters

    # Benchmark compiled mode
    compiled_model = torch.compile(model, mode="max-autotune")

    # Extra warmup for compilation
    for _ in range(num_warmup * 2):
        _ = compiled_model(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = compiled_model(x)
    torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - start) * 1000 / num_iters

    speedup = eager_time / compiled_time

    results["eager_ms"] = eager_time
    results["compiled_ms"] = compiled_time
    results["speedup"] = speedup

    print(f"  Eager:    {eager_time:.3f}ms")
    print(f"  Compiled: {compiled_time:.3f}ms")
    print(f"  Speedup:  {speedup:.2f}x")

    return results


def benchmark_dtype_comparison(
    batch_size: int = 4,
    seq_len: int = 2048,
    d_model: int = 768,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> dict:
    """Benchmark BF16 vs FP32 performance."""
    print_header("DTYPE COMPARISON (BF16 vs FP32)")

    results = {}

    for dtype, name in [(torch.float32, "FP32"), (torch.bfloat16, "BF16")]:
        a = torch.randn(batch_size * seq_len, d_model, device="cuda", dtype=dtype)
        b = torch.randn(d_model, d_model, device="cuda", dtype=dtype)

        # Warmup
        for _ in range(num_warmup):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / num_iters

        # Memory usage
        mem_mb = (a.numel() + b.numel()) * a.element_size() / 1e6

        results[name] = {
            "time_ms": elapsed,
            "memory_mb": mem_mb,
        }

        print(f"  {name}: {elapsed:.3f}ms | {mem_mb:.1f}MB")

    if "FP32" in results and "BF16" in results:
        speedup = results["FP32"]["time_ms"] / results["BF16"]["time_ms"]
        mem_savings = results["FP32"]["memory_mb"] / results["BF16"]["memory_mb"]
        print(f"  BF16 Speedup: {speedup:.2f}x")
        print(f"  BF16 Memory Savings: {mem_savings:.2f}x")

    return results


def run_quick_benchmark():
    """Run quick sanity check benchmarks."""
    print_header("QUICK BENCHMARK")

    from training.hardware import detect_hardware, print_hardware_info
    info = detect_hardware()
    print_hardware_info(info)

    # Quick matmul
    benchmark_matmul(sizes=[(2048, 2048)], num_iters=20)

    # Quick attention
    benchmark_attention(batch_sizes=[4], seq_lengths=[1024], num_iters=10)

    # Quick Mamba
    benchmark_mamba(batch_sizes=[4], seq_lengths=[1024], num_iters=10)

    print("\n✓ Quick benchmark complete!")


def run_full_benchmark():
    """Run full benchmark suite."""
    from training.hardware import detect_hardware, print_hardware_info
    info = detect_hardware()
    print_hardware_info(info)

    results = {}

    results["matmul"] = benchmark_matmul()
    results["attention"] = benchmark_attention()
    results["mamba"] = benchmark_mamba()
    results["memory_bandwidth"] = benchmark_memory_bandwidth()
    results["nvlink"] = benchmark_nvlink()
    results["compile"] = benchmark_compile()
    results["dtype"] = benchmark_dtype_comparison()

    print_header("BENCHMARK COMPLETE")
    print("\nAll benchmarks finished successfully!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Hardware Benchmark for Document-Level NMT")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Run quick sanity check")
    parser.add_argument("--matmul", action="store_true", help="Run matmul benchmark only")
    parser.add_argument("--attention", action="store_true", help="Run attention benchmark only")
    parser.add_argument("--mamba", action="store_true", help="Run Mamba benchmark only")
    parser.add_argument("--memory", action="store_true", help="Run memory bandwidth benchmark only")
    parser.add_argument("--nvlink", action="store_true", help="Run NVLink benchmark only")
    parser.add_argument("--compile", action="store_true", help="Run torch.compile benchmark only")
    parser.add_argument("--dtype", action="store_true", help="Run dtype comparison benchmark only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    # Apply optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.quick:
        run_quick_benchmark()
    elif args.full:
        run_full_benchmark()
    elif args.matmul:
        benchmark_matmul()
    elif args.attention:
        benchmark_attention()
    elif args.mamba:
        benchmark_mamba()
    elif args.memory:
        benchmark_memory_bandwidth()
    elif args.nvlink:
        benchmark_nvlink()
    elif args.compile:
        benchmark_compile()
    elif args.dtype:
        benchmark_dtype_comparison()
    else:
        # Default: quick benchmark
        run_quick_benchmark()


if __name__ == "__main__":
    main()
