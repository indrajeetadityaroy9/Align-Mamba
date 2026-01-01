"""
Length Sensitivity Analysis for Document-Level NMT.

Evaluates model performance and memory usage at different sequence lengths.
Critical for validating Mamba's O(L) complexity advantage over Transformers.

Key tests:
1. Quality at 2x-4x training length
2. Memory scaling behavior
3. Inference speed scaling
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time
import warnings

import torch

from .metrics import EvaluationSuite, EvaluationResult


@dataclass
class LengthAnalysisResult:
    """Results from length sensitivity analysis."""
    by_length: Dict[str, Dict] = field(default_factory=dict)
    memory_scaling: Dict[int, float] = field(default_factory=dict)
    speed_scaling: Dict[int, float] = field(default_factory=dict)
    extrapolation_quality: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        lengths = list(self.by_length.keys())
        return f"LengthAnalysisResult(lengths={lengths})"

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Length Sensitivity Analysis"]
        lines.append("=" * 40)

        for length, metrics in self.by_length.items():
            lines.append(f"\n{length}:")
            if "bleu" in metrics:
                lines.append(f"  BLEU: {metrics['bleu']:.2f}")
            if "comet" in metrics:
                lines.append(f"  COMET: {metrics['comet']:.4f}")
            if "memory_gb" in metrics:
                lines.append(f"  Memory: {metrics['memory_gb']:.2f} GB")
            if "tokens_per_sec" in metrics:
                lines.append(f"  Speed: {metrics['tokens_per_sec']:.1f} tok/s")

        return "\n".join(lines)


class LengthSensitivityAnalyzer:
    """
    Analyzes model performance across different sequence lengths.

    Tests:
    1. Translation quality at various lengths
    2. Memory consumption scaling
    3. Inference speed scaling
    4. Extrapolation to unseen lengths
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_length: int = 8192,
    ):
        """
        Args:
            model: NMT model
            tokenizer: Tokenizer
            device: Device for computation
            max_length: Maximum sequence length to test
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        self.eval_suite = EvaluationSuite(use_comet=True)

    def group_by_length(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        length_buckets: List[int] = [128, 256, 512, 1024, 2048, 4096],
    ) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
        """
        Group examples by source length.

        Args:
            sources: Source texts
            hypotheses: System outputs
            references: Reference translations
            length_buckets: Length thresholds

        Returns:
            Dictionary mapping length ranges to (src, hyp, ref) tuples
        """
        groups = {}
        bucket_names = []

        # Create bucket names
        bucket_names.append(f"<{length_buckets[0]}")
        for i in range(len(length_buckets) - 1):
            bucket_names.append(f"{length_buckets[i]}-{length_buckets[i+1]}")
        bucket_names.append(f">{length_buckets[-1]}")

        # Initialize groups
        for name in bucket_names:
            groups[name] = ([], [], [])

        # Assign to buckets
        for src, hyp, ref in zip(sources, hypotheses, references):
            # Tokenize to get length
            src_tokens = self.tokenizer.encode_source(src, return_tensors=False)
            src_len = len(src_tokens["input_ids"])

            # Find bucket
            bucket_idx = len(length_buckets)  # Default to last bucket
            for i, threshold in enumerate(length_buckets):
                if src_len < threshold:
                    bucket_idx = i
                    break

            bucket_name = bucket_names[bucket_idx]
            groups[bucket_name][0].append(src)
            groups[bucket_name][1].append(hyp)
            groups[bucket_name][2].append(ref)

        # Remove empty buckets
        return {k: v for k, v in groups.items() if v[0]}

    def analyze_quality_by_length(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        length_buckets: List[int] = [128, 256, 512, 1024, 2048, 4096],
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate quality metrics by source length.

        Args:
            sources: Source texts
            hypotheses: System outputs
            references: Reference translations
            length_buckets: Length thresholds

        Returns:
            Dictionary mapping length ranges to EvaluationResult
        """
        groups = self.group_by_length(sources, hypotheses, references, length_buckets)

        results = {}
        for length_range, (srcs, hyps, refs) in groups.items():
            if srcs:
                results[length_range] = self.eval_suite.evaluate(srcs, hyps, refs)

        return results

    @torch.no_grad()
    def measure_memory(
        self,
        batch_size: int = 1,
        seq_length: int = 1024,
    ) -> float:
        """
        Measure GPU memory consumption for given sequence length.

        Args:
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Memory in GB
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create dummy input
        src_ids = torch.randint(
            0, self.tokenizer.vocab_size,
            (batch_size, seq_length),
            device=self.device,
        )
        tgt_ids = torch.randint(
            0, self.tokenizer.vocab_size,
            (batch_size, seq_length // 2),
            device=self.device,
        )

        # Forward pass
        self.model.eval()
        _ = self.model(src_ids, tgt_ids)

        memory_bytes = torch.cuda.max_memory_allocated()
        return memory_bytes / (1024 ** 3)

    @torch.no_grad()
    def measure_speed(
        self,
        batch_size: int = 1,
        seq_length: int = 1024,
        n_warmup: int = 3,
        n_trials: int = 10,
    ) -> float:
        """
        Measure inference speed for given sequence length.

        Args:
            batch_size: Batch size
            seq_length: Sequence length
            n_warmup: Warmup iterations
            n_trials: Measurement iterations

        Returns:
            Tokens per second
        """
        src_ids = torch.randint(
            0, self.tokenizer.vocab_size,
            (batch_size, seq_length),
            device=self.device,
        )
        tgt_ids = torch.randint(
            0, self.tokenizer.vocab_size,
            (batch_size, seq_length // 2),
            device=self.device,
        )

        self.model.eval()

        # Warmup
        for _ in range(n_warmup):
            _ = self.model(src_ids, tgt_ids)
            torch.cuda.synchronize()

        # Measure
        start = time.perf_counter()
        for _ in range(n_trials):
            _ = self.model(src_ids, tgt_ids)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_tokens = n_trials * batch_size * (seq_length + seq_length // 2)
        return total_tokens / elapsed

    def analyze_scaling(
        self,
        lengths: List[int] = [128, 256, 512, 1024, 2048, 4096],
        batch_size: int = 1,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Analyze memory and speed scaling.

        Args:
            lengths: Sequence lengths to test
            batch_size: Batch size

        Returns:
            Tuple of (memory_by_length, speed_by_length)
        """
        memory_scaling = {}
        speed_scaling = {}

        for length in lengths:
            if length > self.max_length:
                continue

            try:
                memory_scaling[length] = self.measure_memory(batch_size, length)
                speed_scaling[length] = self.measure_speed(batch_size, length)
            except RuntimeError as e:
                warnings.warn(f"Failed at length {length}: {e}")
                break

        return memory_scaling, speed_scaling

    def analyze(
        self,
        sources: Optional[List[str]] = None,
        hypotheses: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        test_lengths: List[int] = [128, 256, 512, 1024, 2048],
    ) -> LengthAnalysisResult:
        """
        Run full length sensitivity analysis.

        Args:
            sources: Source texts (optional, for quality analysis)
            hypotheses: System outputs (optional)
            references: Reference translations (optional)
            test_lengths: Lengths for scaling tests

        Returns:
            LengthAnalysisResult
        """
        result = LengthAnalysisResult()

        # Quality analysis by length
        if sources and hypotheses and references:
            quality_by_length = self.analyze_quality_by_length(
                sources, hypotheses, references, test_lengths
            )
            for length_range, eval_result in quality_by_length.items():
                result.by_length[length_range] = eval_result.to_dict()

        # Memory and speed scaling
        memory_scaling, speed_scaling = self.analyze_scaling(test_lengths)
        result.memory_scaling = memory_scaling
        result.speed_scaling = speed_scaling

        # Add to by_length for unified view
        for length in test_lengths:
            length_key = str(length)
            if length_key not in result.by_length:
                result.by_length[length_key] = {}

            if length in memory_scaling:
                result.by_length[length_key]["memory_gb"] = memory_scaling[length]
            if length in speed_scaling:
                result.by_length[length_key]["tokens_per_sec"] = speed_scaling[length]

        return result


class ExtrapolationTester:
    """
    Tests model's ability to extrapolate to longer sequences.

    Mamba should handle longer sequences better than Transformers
    due to its O(L) complexity and recurrent structure.
    """

    def __init__(
        self,
        model,
        tokenizer,
        training_max_length: int = 2048,
        device: str = "cuda",
    ):
        """
        Args:
            model: NMT model
            tokenizer: Tokenizer
            training_max_length: Maximum length seen during training
            device: Device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_max_length = training_max_length
        self.device = device

    @torch.no_grad()
    def test_extrapolation(
        self,
        test_lengths: List[int] = None,
    ) -> Dict[int, Dict]:
        """
        Test model on lengths beyond training.

        Args:
            test_lengths: Lengths to test (default: 1x, 2x, 4x training length)

        Returns:
            Dictionary with results per length
        """
        if test_lengths is None:
            test_lengths = [
                self.training_max_length,
                self.training_max_length * 2,
                self.training_max_length * 4,
            ]

        results = {}
        self.model.eval()

        for length in test_lengths:
            try:
                # Create test input
                src_ids = torch.randint(
                    0, self.tokenizer.vocab_size,
                    (1, length),
                    device=self.device,
                )

                # Try encoding
                start = time.perf_counter()
                encoder_out = self.model.encode(src_ids)
                encode_time = time.perf_counter() - start

                # Try generation
                start = time.perf_counter()
                generated = self.model.generate(src_ids, max_length=50)
                generate_time = time.perf_counter() - start

                results[length] = {
                    "success": True,
                    "encode_time": encode_time,
                    "generate_time": generate_time,
                    "generated_length": generated.shape[1],
                    "relative_to_training": length / self.training_max_length,
                }

            except RuntimeError as e:
                results[length] = {
                    "success": False,
                    "error": str(e),
                    "relative_to_training": length / self.training_max_length,
                }

        return results


def analyze_length_sensitivity(
    model,
    tokenizer,
    sources: Optional[List[str]] = None,
    hypotheses: Optional[List[str]] = None,
    references: Optional[List[str]] = None,
    device: str = "cuda",
) -> LengthAnalysisResult:
    """
    Convenience function for length sensitivity analysis.

    Args:
        model: NMT model
        tokenizer: Tokenizer
        sources: Optional source texts
        hypotheses: Optional system outputs
        references: Optional references
        device: Device

    Returns:
        LengthAnalysisResult
    """
    analyzer = LengthSensitivityAnalyzer(model, tokenizer, device)
    return analyzer.analyze(sources, hypotheses, references)
