#!/usr/bin/env python3
"""
Model Inference Benchmark Script for Document-Level NMT.

Measures model-specific performance metrics:
1. Time-To-First-Token (TTFT) - includes encoder prefill
2. Decoding Throughput (tokens/sec) - per-token generation speed
3. Peak Memory Usage - cache sizes for different architectures
4. End-to-End Latency - TTFT + (num_tokens × decode_time)

Compares:
- Pure Mamba (Decoder-Only Concatenative)
- Hybrid Mamba-Attention (Our model)
- Transformer baseline (if available)

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --model hybrid --checkpoint path/to/ckpt
    python scripts/benchmark_inference.py --synthetic  # Use synthetic data
    python scripts/benchmark_inference.py --compare-all  # Run all model types
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model_name: str
    batch_size: int
    src_length: int
    tgt_length: int

    # Time-To-First-Token (includes encoder prefill)
    ttft_ms: float = 0.0
    ttft_std_ms: float = 0.0

    # Decoding throughput
    decode_tokens_per_sec: float = 0.0
    decode_time_per_token_ms: float = 0.0

    # End-to-end latency
    e2e_latency_ms: float = 0.0

    # Memory usage
    peak_memory_mb: float = 0.0
    encoder_memory_mb: float = 0.0
    decoder_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0

    # Throughput summary
    total_tokens_per_sec: float = 0.0

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.model_name}):\n"
            f"  TTFT: {self.ttft_ms:.2f}ms (±{self.ttft_std_ms:.2f})\n"
            f"  Decode: {self.decode_tokens_per_sec:.1f} tok/s ({self.decode_time_per_token_ms:.3f}ms/tok)\n"
            f"  E2E Latency: {self.e2e_latency_ms:.2f}ms\n"
            f"  Peak Memory: {self.peak_memory_mb:.1f}MB\n"
            f"  Total Throughput: {self.total_tokens_per_sec:.1f} tok/s"
        )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    src_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    tgt_lengths: List[int] = field(default_factory=lambda: [64, 128, 256])

    num_warmup: int = 3
    num_iterations: int = 10

    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


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


class InferenceBenchmarker:
    """
    Benchmarker for NMT model inference performance.

    Measures:
    - TTFT (Time-To-First-Token): Encoder prefill + first decoder step
    - Decoding throughput: Tokens/sec during autoregressive generation
    - Memory usage: Peak, encoder, decoder, and cache memory
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def benchmark_encoder_prefill(
        self,
        model: nn.Module,
        src_ids: torch.Tensor,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> Tuple[float, float]:
        """
        Benchmark encoder prefill time.

        Returns:
            Tuple of (mean_time_ms, std_time_ms)
        """
        model.eval()
        times = []

        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                if hasattr(model, 'encode'):
                    _ = model.encode(src_ids)
                elif hasattr(model, 'encoder'):
                    _ = model.encoder(src_ids)
                else:
                    # Assume full forward
                    bos = torch.full((src_ids.size(0), 1), 1, dtype=torch.long, device=src_ids.device)
                    _ = model(src_ids, bos)
            torch.cuda.synchronize()

            # Benchmark
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()

                if hasattr(model, 'encode'):
                    _ = model.encode(src_ids)
                elif hasattr(model, 'encoder'):
                    _ = model.encoder(src_ids)
                else:
                    bos = torch.full((src_ids.size(0), 1), 1, dtype=torch.long, device=src_ids.device)
                    _ = model(src_ids, bos)

                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

        return mean_time, std_time

    def benchmark_ttft(
        self,
        model: nn.Module,
        src_ids: torch.Tensor,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> Tuple[float, float]:
        """
        Benchmark Time-To-First-Token.

        This includes:
        - Encoder prefill (for seq2seq models)
        - Cache initialization
        - First decoder step

        Returns:
            Tuple of (mean_ttft_ms, std_ttft_ms)
        """
        model.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        times = []

        bos_token_id = getattr(model.config, 'bos_token_id', 1) if hasattr(model, 'config') else 1

        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                # Encode
                if hasattr(model, 'encode'):
                    encoder_out = model.encode(src_ids)
                    # Initialize cache
                    if hasattr(model, 'init_generation_cache'):
                        cache = model.init_generation_cache(encoder_out)
                    # First decoder step
                    bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                    if hasattr(model, 'generate_step'):
                        _ = model.generate_step(bos, cache)
                else:
                    # Decoder-only: process source + first target token
                    bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                    _ = model(src_ids, bos)
            torch.cuda.synchronize()

            # Benchmark
            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()

                # Encode + first decode step
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
        """
        Benchmark autoregressive decoding throughput.

        Returns:
            Tuple of (tokens_per_sec, time_per_token_ms)
        """
        model.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device

        bos_token_id = getattr(model.config, 'bos_token_id', 1) if hasattr(model, 'config') else 1

        decode_times = []

        with torch.no_grad():
            # Setup: encode once (not counted in decode time)
            if hasattr(model, 'encode'):
                encoder_out = model.encode(src_ids)
                use_cache = hasattr(model, 'init_generation_cache')
            else:
                encoder_out = None
                use_cache = False

            # Warmup
            for _ in range(num_warmup):
                if use_cache:
                    cache = model.init_generation_cache(encoder_out)
                    token = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                    for _ in range(min(num_decode_tokens, 16)):  # Shorter warmup
                        _, cache = model.generate_step(token, cache)
                        token = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                else:
                    # Full forward (no incremental decoding)
                    tgt = torch.randint(1, 100, (batch_size, num_decode_tokens), device=device)
                    if encoder_out is not None:
                        _ = model.decoder(tgt, encoder_out)
                    else:
                        _ = model(src_ids, tgt)
            torch.cuda.synchronize()

            # Benchmark decoding
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
                else:
                    tgt = torch.randint(1, 100, (batch_size, num_decode_tokens), device=device)

                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    if encoder_out is not None:
                        _ = model.decoder(tgt, encoder_out)
                    else:
                        _ = model(src_ids, tgt)

                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    decode_times.append(elapsed)

        mean_decode_time = sum(decode_times) / len(decode_times)
        total_tokens = batch_size * num_decode_tokens

        tokens_per_sec = total_tokens / mean_decode_time
        time_per_token_ms = (mean_decode_time / num_decode_tokens) * 1000

        return tokens_per_sec, time_per_token_ms

    def benchmark_memory(
        self,
        model: nn.Module,
        src_ids: torch.Tensor,
        num_decode_tokens: int = 64,
    ) -> Dict[str, float]:
        """
        Benchmark memory usage during inference.

        Returns:
            Dict with memory breakdown in MB
        """
        model.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device

        bos_token_id = getattr(model.config, 'bos_token_id', 1) if hasattr(model, 'config') else 1

        reset_memory_stats()
        baseline_memory = get_memory_usage()

        with torch.no_grad():
            # Measure encoder memory
            if hasattr(model, 'encode'):
                encoder_out = model.encode(src_ids)
                torch.cuda.synchronize()
                encoder_memory = get_memory_usage() - baseline_memory

                # Measure cache initialization memory
                if hasattr(model, 'init_generation_cache'):
                    cache = model.init_generation_cache(encoder_out)
                    torch.cuda.synchronize()
                    cache_memory = get_memory_usage() - baseline_memory - encoder_memory
                else:
                    cache_memory = 0.0

                # Measure decoder memory during generation
                token = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
                for _ in range(num_decode_tokens):
                    if hasattr(model, 'generate_step') and 'cache' in dir():
                        _, cache = model.generate_step(token, cache)
                    else:
                        break
                torch.cuda.synchronize()
                decoder_memory = get_memory_usage() - baseline_memory - encoder_memory
            else:
                # Decoder-only
                encoder_memory = 0.0
                cache_memory = 0.0
                tgt = torch.randint(1, 100, (batch_size, num_decode_tokens), device=device)
                _ = model(src_ids, tgt)
                torch.cuda.synchronize()
                decoder_memory = get_memory_usage() - baseline_memory

        peak_memory = get_peak_memory()

        return {
            "peak_memory_mb": peak_memory,
            "encoder_memory_mb": encoder_memory,
            "decoder_memory_mb": decoder_memory,
            "cache_memory_mb": cache_memory,
        }

    def run_benchmark(
        self,
        model: nn.Module,
        model_name: str,
        batch_size: int = 1,
        src_length: int = 512,
        tgt_length: int = 128,
    ) -> BenchmarkResult:
        """
        Run full benchmark for a model configuration.

        Args:
            model: The NMT model to benchmark
            model_name: Name for logging
            batch_size: Batch size
            src_length: Source sequence length
            tgt_length: Target sequence length (number of decode tokens)

        Returns:
            BenchmarkResult with all metrics
        """
        device = self.config.device

        # Create synthetic input
        src_ids = torch.randint(10, 1000, (batch_size, src_length), device=device)

        # Run benchmarks
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

        memory = self.benchmark_memory(model, src_ids, num_decode_tokens=tgt_length)

        # Calculate end-to-end latency
        e2e_latency = ttft_ms + (tgt_length * time_per_token)

        # Total throughput (including prefill amortized over sequence)
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
            peak_memory_mb=memory["peak_memory_mb"],
            encoder_memory_mb=memory["encoder_memory_mb"],
            decoder_memory_mb=memory["decoder_memory_mb"],
            cache_memory_mb=memory["cache_memory_mb"],
            total_tokens_per_sec=total_throughput,
        )

    def run_sweep(
        self,
        model: nn.Module,
        model_name: str,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark sweep across all configurations.

        Returns:
            List of BenchmarkResult for each configuration
        """
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
            use_hybrid_blocks=True,
        )

        model = HybridMambaEncoderDecoder(
            config=config,
            device=device,
            dtype=dtype,
        )
        model.eval()
        return model

    except ImportError as e:
        print(f"Could not import Hybrid model: {e}")
        return None


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


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


def run_model_comparison(config: BenchmarkConfig):
    """Compare different model architectures."""
    print_header("MODEL COMPARISON BENCHMARK")

    all_results = {}

    # Benchmark Hybrid model
    print("\n[1/3] Creating Hybrid Mamba-Attention model...")
    hybrid_model = create_hybrid_model(device=config.device, dtype=config.dtype)

    if hybrid_model is not None:
        print(f"  Model params: {hybrid_model.num_parameters() / 1e6:.1f}M")
        benchmarker = InferenceBenchmarker(config)
        results = benchmarker.run_sweep(hybrid_model, "Hybrid")
        all_results["Hybrid"] = results
        print_results_table(results)
        del hybrid_model
        torch.cuda.empty_cache()

    # Summary comparison
    print_header("SUMMARY")

    for model_name, results in all_results.items():
        if results:
            avg_ttft = sum(r.ttft_ms for r in results) / len(results)
            avg_decode = sum(r.decode_tokens_per_sec for r in results) / len(results)
            avg_memory = sum(r.peak_memory_mb for r in results) / len(results)
            print(f"{model_name}:")
            print(f"  Avg TTFT: {avg_ttft:.2f}ms")
            print(f"  Avg Decode: {avg_decode:.0f} tok/s")
            print(f"  Avg Peak Memory: {avg_memory:.0f}MB")


def run_quick_benchmark():
    """Run a quick sanity check benchmark."""
    print_header("QUICK INFERENCE BENCHMARK")

    config = BenchmarkConfig(
        batch_sizes=[1, 4],
        src_lengths=[512],
        tgt_lengths=[64],
        num_warmup=2,
        num_iterations=5,
    )

    # Create model
    print("\nCreating Hybrid model...")
    model = create_hybrid_model(device=config.device, dtype=config.dtype)

    if model is None:
        print("Could not create model. Exiting.")
        return

    print(f"Model params: {model.num_parameters() / 1e6:.1f}M")

    # Run benchmark
    benchmarker = InferenceBenchmarker(config)
    results = benchmarker.run_sweep(model, "Hybrid")

    print_results_table(results)

    print("\n Quick benchmark complete!")


def main():
    parser = argparse.ArgumentParser(description="Model Inference Benchmark for Document-Level NMT")
    parser.add_argument("--quick", action="store_true", help="Run quick sanity check")
    parser.add_argument("--full", action="store_true", help="Run full benchmark sweep")
    parser.add_argument("--compare-all", action="store_true", help="Compare all model types")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8],
                       help="Batch sizes to benchmark")
    parser.add_argument("--src-lengths", type=int, nargs="+", default=[256, 512, 1024, 2048],
                       help="Source lengths to benchmark")
    parser.add_argument("--tgt-lengths", type=int, nargs="+", default=[64, 128, 256],
                       help="Target lengths to benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Inference benchmarks require GPU.")
        return

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Apply optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        src_lengths=args.src_lengths,
        tgt_lengths=args.tgt_lengths,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
    )

    if args.quick:
        run_quick_benchmark()
    elif args.compare_all:
        run_model_comparison(config)
    elif args.full:
        run_model_comparison(config)
    else:
        run_quick_benchmark()


if __name__ == "__main__":
    main()
