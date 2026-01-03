#!/usr/bin/env python3
"""
Unified Benchmark Script for Document-Level NMT.

Combines hardware and inference benchmarks into a single interface.

Usage:
    python scripts/benchmark.py --hardware     # GPU compute benchmarks (matmul, attention, Mamba)
    python scripts/benchmark.py --inference    # Model inference benchmarks (TTFT, decoding, latency)
    python scripts/benchmark.py --all          # Both hardware and inference benchmarks
    python scripts/benchmark.py --quick        # Quick sanity check (both)

Hardware Benchmarks:
    - Matmul performance (TFLOPS)
    - Attention (Flash SDP) throughput
    - Mamba-2 throughput
    - Memory bandwidth
    - NVLink (multi-GPU)
    - torch.compile speedup
    - BF16 vs FP32 comparison

Inference Benchmarks:
    - Time-To-First-Token (TTFT)
    - Decoding throughput (tokens/sec)
    - End-to-end latency
    - Peak memory usage
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


# =============================================================================
# Utilities
# =============================================================================

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def get_peak_memory() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


# =============================================================================
# Hardware Benchmarks
# =============================================================================

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
        k = n
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)

        for _ in range(num_warmup):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        flops = 2 * m * n * k * num_iters
        tflops = flops / elapsed / 1e12

        results[f"{m}x{k}x{n}"] = {"time_ms": elapsed * 1000 / num_iters, "tflops": tflops}
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

                for _ in range(num_warmup):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(num_iters):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                time_ms = elapsed * 1000 / num_iters
                tokens_per_sec = batch_size * seq_len / (elapsed / num_iters)

                results[f"B{batch_size}_L{seq_len}"] = {"time_ms": time_ms, "tokens_per_sec": tokens_per_sec}
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
    mamba = Mamba2(d_model=d_model, d_state=d_state, device="cuda", dtype=dtype)
    mamba.eval()

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            try:
                x = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=dtype)

                with torch.no_grad():
                    for _ in range(num_warmup):
                        _ = mamba(x)
                torch.cuda.synchronize()

                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(num_iters):
                        _ = mamba(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                time_ms = elapsed * 1000 / num_iters
                tokens_per_sec = batch_size * seq_len / (elapsed / num_iters)

                results[f"B{batch_size}_L{seq_len}"] = {"time_ms": time_ms, "tokens_per_sec": tokens_per_sec}
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

            for _ in range(5):
                b.copy_(a)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(num_iters):
                b.copy_(a)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bytes_transferred = 2 * numel * bytes_per_elem * num_iters
            bandwidth_gbps = bytes_transferred / elapsed / 1e9

            results[f"{size_gb}GB"] = {"bandwidth_gbps": bandwidth_gbps}
            print(f"  [{size_gb}GB tensor] {bandwidth_gbps:.1f} GB/s")

            del a, b
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  [{size_gb}GB tensor] OOM")
            torch.cuda.empty_cache()

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
    model = SimpleMLP(d_model).cuda().bfloat16()
    x = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.bfloat16)

    model.eval()
    for _ in range(num_warmup):
        _ = model(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = model(x)
    torch.cuda.synchronize()
    eager_time = (time.perf_counter() - start) * 1000 / num_iters

    compiled_model = torch.compile(model, mode="max-autotune")

    for _ in range(num_warmup * 2):
        _ = compiled_model(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = compiled_model(x)
    torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - start) * 1000 / num_iters

    speedup = eager_time / compiled_time

    results = {"eager_ms": eager_time, "compiled_ms": compiled_time, "speedup": speedup}
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

        for _ in range(num_warmup):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / num_iters

        mem_mb = (a.numel() + b.numel()) * a.element_size() / 1e6

        results[name] = {"time_ms": elapsed, "memory_mb": mem_mb}
        print(f"  {name}: {elapsed:.3f}ms | {mem_mb:.1f}MB")

    if "FP32" in results and "BF16" in results:
        speedup = results["FP32"]["time_ms"] / results["BF16"]["time_ms"]
        mem_savings = results["FP32"]["memory_mb"] / results["BF16"]["memory_mb"]
        print(f"  BF16 Speedup: {speedup:.2f}x")
        print(f"  BF16 Memory Savings: {mem_savings:.2f}x")

    return results


# =============================================================================
# Inference Benchmarks
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single inference benchmark run."""
    model_name: str
    batch_size: int
    src_length: int
    tgt_length: int
    ttft_ms: float = 0.0
    ttft_std_ms: float = 0.0
    decode_tokens_per_sec: float = 0.0
    decode_time_per_token_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    total_tokens_per_sec: float = 0.0


@dataclass
class InferenceBenchmarkConfig:
    """Configuration for inference benchmarks."""
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    src_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    tgt_lengths: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_warmup: int = 3
    num_iterations: int = 10
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


class InferenceBenchmarker:
    """Benchmarker for NMT model inference performance."""

    def __init__(self, config: Optional[InferenceBenchmarkConfig] = None):
        self.config = config or InferenceBenchmarkConfig()

    def benchmark_ttft(
        self,
        model: nn.Module,
        src_ids: torch.Tensor,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> Tuple[float, float]:
        """Benchmark Time-To-First-Token."""
        model.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        times = []

        bos_token_id = getattr(model.config, 'bos_token_id', 1) if hasattr(model, 'config') else 1

        with torch.no_grad():
            for _ in range(num_warmup):
                if hasattr(model, 'encode'):
                    encoder_out = model.encode(src_ids)
                    if hasattr(model, 'init_generation_cache'):
                        cache = model.init_generation_cache(encoder_out)
                    bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                    if hasattr(model, 'generate_step'):
                        _ = model.generate_step(bos, cache)
                else:
                    bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                    _ = model(src_ids, bos)
            torch.cuda.synchronize()

            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()

                if hasattr(model, 'encode'):
                    encoder_out = model.encode(src_ids)
                    if hasattr(model, 'init_generation_cache'):
                        cache = model.init_generation_cache(encoder_out)
                    bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                    if hasattr(model, 'generate_step'):
                        _ = model.generate_step(bos, cache)
                else:
                    bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                    _ = model(src_ids, bos)

                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

        return mean_time, std_time

    def benchmark_decoding(
        self,
        model: nn.Module,
        src_ids: torch.Tensor,
        num_decode_tokens: int = 64,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> Tuple[float, float]:
        """Benchmark autoregressive decoding throughput."""
        model.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device

        bos_token_id = getattr(model.config, 'bos_token_id', 1) if hasattr(model, 'config') else 1

        decode_times = []

        with torch.no_grad():
            if hasattr(model, 'encode'):
                encoder_out = model.encode(src_ids)
                use_cache = hasattr(model, 'init_generation_cache')
            else:
                encoder_out = None
                use_cache = False

            for _ in range(num_warmup):
                if use_cache:
                    cache = model.init_generation_cache(encoder_out)
                    token = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                    for _ in range(min(num_decode_tokens, 16)):
                        _, cache = model.generate_step(token, cache)
                        token = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
            torch.cuda.synchronize()

            for _ in range(num_iterations):
                if use_cache:
                    cache = model.init_generation_cache(encoder_out)
                    token = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    for _ in range(num_decode_tokens):
                        _, cache = model.generate_step(token, cache)
                        token = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    decode_times.append(elapsed)

        if not decode_times:
            return 0.0, 0.0

        mean_decode_time = sum(decode_times) / len(decode_times)
        total_tokens = batch_size * num_decode_tokens

        tokens_per_sec = total_tokens / mean_decode_time
        time_per_token_ms = (mean_decode_time / num_decode_tokens) * 1000

        return tokens_per_sec, time_per_token_ms

    def run_benchmark(
        self,
        model: nn.Module,
        model_name: str,
        batch_size: int = 1,
        src_length: int = 512,
        tgt_length: int = 128,
    ) -> BenchmarkResult:
        """Run full inference benchmark for a model configuration."""
        device = self.config.device
        src_ids = torch.randint(10, 1000, (batch_size, src_length), device=device)

        reset_memory_stats()

        ttft_ms, ttft_std = self.benchmark_ttft(
            model, src_ids,
            num_warmup=self.config.num_warmup,
            num_iterations=self.config.num_iterations
        )

        tokens_per_sec, time_per_token = self.benchmark_decoding(
            model, src_ids,
            num_decode_tokens=tgt_length,
            num_warmup=self.config.num_warmup,
            num_iterations=self.config.num_iterations
        )

        peak_memory = get_peak_memory()
        e2e_latency = ttft_ms + (tgt_length * time_per_token)

        total_time_sec = (ttft_ms + tgt_length * time_per_token) / 1000
        total_tokens = batch_size * tgt_length
        total_throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0

        return BenchmarkResult(
            model_name=model_name,
            batch_size=batch_size,
            src_length=src_length,
            tgt_length=tgt_length,
            ttft_ms=ttft_ms,
            ttft_std_ms=ttft_std,
            decode_tokens_per_sec=tokens_per_sec,
            decode_time_per_token_ms=time_per_token,
            e2e_latency_ms=e2e_latency,
            peak_memory_mb=peak_memory,
            total_tokens_per_sec=total_throughput,
        )

    def run_sweep(self, model: nn.Module, model_name: str) -> List[BenchmarkResult]:
        """Run benchmark sweep across all configurations."""
        results = []

        for batch_size in self.config.batch_sizes:
            for src_length in self.config.src_lengths:
                for tgt_length in self.config.tgt_lengths:
                    try:
                        result = self.run_benchmark(
                            model, model_name,
                            batch_size, src_length, tgt_length
                        )
                        results.append(result)
                        print(f"  [B={batch_size}, Src={src_length}, Tgt={tgt_length}] "
                              f"TTFT={result.ttft_ms:.2f}ms, "
                              f"Decode={result.decode_tokens_per_sec:.0f} tok/s, "
                              f"Memory={result.peak_memory_mb:.0f}MB")
                    except torch.cuda.OutOfMemoryError:
                        print(f"  [B={batch_size}, Src={src_length}, Tgt={tgt_length}] OOM")
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"  [B={batch_size}, Src={src_length}, Tgt={tgt_length}] Error: {e}")

        return results


def create_hybrid_model(
    vocab_size: int = 32000,
    d_model: int = 768,
    encoder_layers: int = 16,
    decoder_layers: int = 24,
    d_state: int = 128,
    n_heads: int = 12,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Create the Hybrid Mamba-Attention model."""
    try:
        from models import HybridMambaEncoderDecoder, ModelConfig

        config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            d_state=d_state,
            n_heads=n_heads,
            hybrid_interval=8,
        )

        model = HybridMambaEncoderDecoder(config=config, device=device, dtype=dtype)
        model.eval()
        return model

    except ImportError as e:
        print(f"Could not import Hybrid model: {e}")
        return None


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "-" * 90)
    print(f"{'Config':<25} {'TTFT(ms)':<12} {'Decode(tok/s)':<15} {'E2E(ms)':<12} {'Memory(MB)':<12}")
    print("-" * 90)

    for r in results:
        config = f"B={r.batch_size}, S={r.src_length}, T={r.tgt_length}"
        print(f"{config:<25} {r.ttft_ms:<12.2f} {r.decode_tokens_per_sec:<15.0f} "
              f"{r.e2e_latency_ms:<12.2f} {r.peak_memory_mb:<12.0f}")

    print("-" * 90)


# =============================================================================
# Main Entry Points
# =============================================================================

def run_hardware_benchmarks(quick: bool = False):
    """Run all hardware benchmarks."""
    print_header("HARDWARE BENCHMARKS")

    try:
        from training.hardware import detect_hardware, print_hardware_info
        info = detect_hardware()
        print_hardware_info(info)
    except ImportError:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    if quick:
        results["matmul"] = benchmark_matmul(sizes=[(2048, 2048)], num_iters=20)
        results["attention"] = benchmark_attention(batch_sizes=[4], seq_lengths=[1024], num_iters=10)
        results["mamba"] = benchmark_mamba(batch_sizes=[4], seq_lengths=[1024], num_iters=10)
    else:
        results["matmul"] = benchmark_matmul()
        results["attention"] = benchmark_attention()
        results["mamba"] = benchmark_mamba()
        results["memory_bandwidth"] = benchmark_memory_bandwidth()
        results["compile"] = benchmark_compile()
        results["dtype"] = benchmark_dtype_comparison()

    return results


def run_inference_benchmarks(quick: bool = False):
    """Run inference benchmarks on the Hybrid model."""
    print_header("INFERENCE BENCHMARKS")

    if quick:
        config = InferenceBenchmarkConfig(
            batch_sizes=[1, 4],
            src_lengths=[512],
            tgt_lengths=[64],
            num_warmup=2,
            num_iterations=5,
        )
    else:
        config = InferenceBenchmarkConfig()

    print("\nCreating Hybrid model...")
    model = create_hybrid_model(device=config.device, dtype=config.dtype)

    if model is None:
        print("Could not create model. Skipping inference benchmarks.")
        return {}

    print(f"Model params: {model.num_parameters() / 1e6:.1f}M")

    benchmarker = InferenceBenchmarker(config)
    results = benchmarker.run_sweep(model, "Hybrid")
    print_results_table(results)

    del model
    torch.cuda.empty_cache()

    return {"inference": results}


def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark for Document-Level NMT")
    parser.add_argument("--hardware", action="store_true", help="Run hardware benchmarks only")
    parser.add_argument("--inference", action="store_true", help="Run inference benchmarks only")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick sanity check")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Benchmarks require GPU.")
        return

    # Apply optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    if args.quick:
        run_hardware_benchmarks(quick=True)
        run_inference_benchmarks(quick=True)
    elif args.hardware:
        run_hardware_benchmarks(quick=False)
    elif args.inference:
        run_inference_benchmarks(quick=False)
    elif args.all:
        run_hardware_benchmarks(quick=False)
        run_inference_benchmarks(quick=False)
    else:
        # Default: quick benchmarks
        run_hardware_benchmarks(quick=True)
        run_inference_benchmarks(quick=True)

    print_header("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
