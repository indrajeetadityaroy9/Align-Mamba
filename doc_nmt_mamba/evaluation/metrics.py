"""
Evaluation Metrics for Document-Level NMT.

Provides:
- BLEU (sacrebleu): Standard MT metric
- COMET (wmt22-comet-da): Neural evaluation metric
- Combined evaluation suite
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
import warnings

import torch
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

# COMET is optional (requires additional install)
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    warnings.warn("COMET not available. Install with: pip install unbabel-comet")


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    bleu: float = 0.0
    bleu_details: Dict = field(default_factory=dict)
    chrf: float = 0.0
    ter: float = 0.0
    comet: float = 0.0
    comet_scores: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(BLEU={self.bleu:.2f}, "
            f"chrF={self.chrf:.2f}, TER={self.ter:.2f}, "
            f"COMET={self.comet:.4f})"
        )

    def to_dict(self) -> Dict:
        return {
            "bleu": self.bleu,
            "bleu_details": self.bleu_details,
            "chrf": self.chrf,
            "ter": self.ter,
            "comet": self.comet,
        }


class BLEUScorer:
    """
    BLEU scorer using sacrebleu.

    Standard MT evaluation metric based on n-gram precision.
    """

    def __init__(
        self,
        lowercase: bool = False,
        tokenize: str = "13a",  # Standard tokenization
        smooth_method: str = "exp",
    ):
        """
        Args:
            lowercase: Lowercase before scoring
            tokenize: Tokenization method
            smooth_method: Smoothing for sentence-level BLEU
        """
        self.lowercase = lowercase
        self.tokenize = tokenize
        self.smooth_method = smooth_method

        self.bleu = BLEU(
            lowercase=lowercase,
            tokenize=tokenize,
            smooth_method=smooth_method,
        )

    def score(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Compute corpus-level BLEU.

        Args:
            hypotheses: List of system outputs
            references: List of reference lists (supports multiple refs)

        Returns:
            Dictionary with BLEU score and details
        """
        # sacrebleu expects references as list of lists
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]

        # Transpose for sacrebleu format
        refs_transposed = list(zip(*references))

        result = self.bleu.corpus_score(hypotheses, refs_transposed)

        return {
            "score": result.score,
            "precisions": result.precisions,
            "bp": result.bp,
            "ratio": result.ratio,
            "hyp_len": result.sys_len,
            "ref_len": result.ref_len,
        }

    def sentence_score(
        self,
        hypothesis: str,
        references: List[str],
    ) -> float:
        """Compute sentence-level BLEU."""
        result = self.bleu.sentence_score(hypothesis, references)
        return result.score


class CHRFScorer:
    """
    chrF scorer - character-level metric.

    More robust to morphological variations than BLEU.
    """

    def __init__(
        self,
        char_order: int = 6,
        word_order: int = 0,
        beta: int = 2,
    ):
        self.chrf = CHRF(
            char_order=char_order,
            word_order=word_order,
            beta=beta,
        )

    def score(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> float:
        """Compute corpus-level chrF."""
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]

        refs_transposed = list(zip(*references))
        result = self.chrf.corpus_score(hypotheses, refs_transposed)
        return result.score


class TERScorer:
    """
    Translation Edit Rate scorer.

    Measures minimum edit operations needed to transform hypothesis to reference.
    """

    def __init__(self, normalized: bool = True):
        self.ter = TER(normalized=normalized)

    def score(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> float:
        """Compute corpus-level TER."""
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]

        refs_transposed = list(zip(*references))
        result = self.ter.corpus_score(hypotheses, refs_transposed)
        return result.score


class COMETScorer:
    """
    COMET neural evaluation metric.

    Uses pretrained model to predict human quality judgments.
    Much better correlation with human evaluation than BLEU.
    """

    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 32,
        gpus: int = 1,
    ):
        """
        Args:
            model_name: COMET model to use
            batch_size: Batch size for inference
            gpus: Number of GPUs (0 for CPU)
        """
        if not COMET_AVAILABLE:
            raise ImportError("COMET not available. Install with: pip install unbabel-comet")

        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus

        # Download and load model
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)

        if gpus > 0 and torch.cuda.is_available():
            self.model = self.model.cuda()

    def score(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Compute COMET scores.

        Args:
            sources: Source sentences
            hypotheses: System outputs
            references: Reference translations

        Returns:
            Dictionary with system score and per-sentence scores
        """
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]

        output = self.model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self.gpus,
        )

        return {
            "system_score": output.system_score,
            "scores": output.scores,
        }


class EvaluationSuite:
    """
    Combined evaluation suite for NMT.

    Runs multiple metrics and aggregates results.
    """

    def __init__(
        self,
        use_comet: bool = True,
        comet_model: str = "Unbabel/wmt22-comet-da",
        comet_batch_size: int = 32,
    ):
        """
        Args:
            use_comet: Whether to compute COMET (slower but more accurate)
            comet_model: COMET model name
            comet_batch_size: Batch size for COMET
        """
        self.bleu_scorer = BLEUScorer()
        self.chrf_scorer = CHRFScorer()
        self.ter_scorer = TERScorer()

        self.use_comet = use_comet and COMET_AVAILABLE
        self.comet_scorer = None

        if self.use_comet:
            try:
                self.comet_scorer = COMETScorer(
                    model_name=comet_model,
                    batch_size=comet_batch_size,
                )
            except Exception as e:
                warnings.warn(f"Failed to load COMET: {e}")
                self.use_comet = False

    def evaluate(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> EvaluationResult:
        """
        Run full evaluation suite.

        Args:
            sources: Source sentences
            hypotheses: System outputs
            references: Reference translations

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult()

        # BLEU
        bleu_result = self.bleu_scorer.score(hypotheses, references)
        result.bleu = bleu_result["score"]
        result.bleu_details = bleu_result

        # chrF
        result.chrf = self.chrf_scorer.score(hypotheses, references)

        # TER
        result.ter = self.ter_scorer.score(hypotheses, references)

        # COMET
        if self.use_comet and self.comet_scorer:
            comet_result = self.comet_scorer.score(sources, hypotheses, references)
            result.comet = comet_result["system_score"]
            result.comet_scores = comet_result["scores"]

        return result

    def evaluate_by_length(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        length_buckets: List[int] = [50, 100, 200, 500, 1000],
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate by source length buckets.

        Args:
            sources: Source sentences
            hypotheses: System outputs
            references: Reference translations
            length_buckets: Length thresholds for buckets

        Returns:
            Dictionary mapping bucket names to results
        """
        results = {}

        # Group by length
        buckets = {f"<{length_buckets[0]}": ([], [], [])}
        for i in range(len(length_buckets) - 1):
            buckets[f"{length_buckets[i]}-{length_buckets[i+1]}"] = ([], [], [])
        buckets[f">{length_buckets[-1]}"] = ([], [], [])

        for src, hyp, ref in zip(sources, hypotheses, references):
            src_len = len(src.split())

            bucket_name = f">{length_buckets[-1]}"
            for i, threshold in enumerate(length_buckets):
                if src_len < threshold:
                    if i == 0:
                        bucket_name = f"<{threshold}"
                    else:
                        bucket_name = f"{length_buckets[i-1]}-{threshold}"
                    break

            buckets[bucket_name][0].append(src)
            buckets[bucket_name][1].append(hyp)
            buckets[bucket_name][2].append(ref)

        # Evaluate each bucket
        for bucket_name, (srcs, hyps, refs) in buckets.items():
            if srcs:
                results[bucket_name] = self.evaluate(srcs, hyps, refs)
            else:
                results[bucket_name] = EvaluationResult()

        return results


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Quick BLEU computation."""
    scorer = BLEUScorer()
    return scorer.score(hypotheses, references)["score"]


def compute_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Quick COMET computation."""
    if not COMET_AVAILABLE:
        warnings.warn("COMET not available")
        return 0.0

    scorer = COMETScorer()
    return scorer.score(sources, hypotheses, references)["system_score"]
