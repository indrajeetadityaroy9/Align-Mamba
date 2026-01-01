"""
Contrastive Pronoun Evaluation for Document-Level NMT.

Based on ContraPro (Müller et al., 2018) - evaluates anaphoric pronoun translation.
This is the KEY metric for document-level NMT quality.

Target: +10pp improvement over sentence-level baseline.
"""

from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class ContrastiveExample:
    """A contrastive evaluation example."""
    source_context: str  # Source with antecedent
    source_current: str  # Current source sentence
    correct_translation: str  # Correct target with pronoun
    contrastive_translations: List[str]  # Wrong pronoun alternatives
    pronoun_type: str  # "he/she/it/they"
    antecedent_distance: int  # Distance in sentences
    antecedent: str  # The antecedent noun phrase
    example_id: str = ""


@dataclass
class ContrastiveResult:
    """Results from contrastive evaluation."""
    accuracy: float = 0.0
    total_examples: int = 0
    correct: int = 0
    by_pronoun_type: Dict[str, float] = field(default_factory=dict)
    by_distance: Dict[int, float] = field(default_factory=dict)
    examples_evaluated: List[Dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ContrastiveResult(accuracy={self.accuracy:.2%}, "
            f"correct={self.correct}/{self.total_examples})"
        )


class ContrastivePronounEvaluator:
    """
    Contrastive evaluation for anaphoric pronouns.

    Method:
    1. Given source with context (containing antecedent)
    2. Compare model scores for correct vs incorrect pronoun translations
    3. Model is "correct" if it assigns higher probability to correct pronoun

    This tests whether the model properly uses document context to resolve
    pronoun ambiguity.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        """
        Args:
            model: NMT model with scoring capability
            tokenizer: Tokenizer for model
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def score_translation(
        self,
        source: str,
        target: str,
    ) -> float:
        """
        Compute log probability of target given source.

        Args:
            source: Source text
            target: Target text

        Returns:
            Log probability (higher = more likely)
        """
        # Encode
        src_ids, tgt_ids = self.tokenizer.encode_pair(source, target)
        src_ids = src_ids.unsqueeze(0).to(self.device)
        tgt_ids = tgt_ids.unsqueeze(0).to(self.device)

        # Get model output
        self.model.eval()
        logits = self.model(src_ids, tgt_ids[:, :-1])

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log prob of actual target tokens
        target_tokens = tgt_ids[:, 1:]  # Shift for teacher forcing
        token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)

        # Sum log probs (or average for length normalization)
        total_log_prob = token_log_probs.sum().item()
        avg_log_prob = token_log_probs.mean().item()

        return avg_log_prob  # Use average for length normalization

    def evaluate_example(
        self,
        example: ContrastiveExample,
    ) -> Tuple[bool, Dict]:
        """
        Evaluate a single contrastive example.

        Args:
            example: Contrastive example with correct and wrong translations

        Returns:
            Tuple of (is_correct, details_dict)
        """
        # Combine context and current sentence
        full_source = f"{example.source_context} {example.source_current}"

        # Score correct translation
        correct_score = self.score_translation(full_source, example.correct_translation)

        # Score contrastive translations
        contrastive_scores = [
            self.score_translation(full_source, trans)
            for trans in example.contrastive_translations
        ]

        # Model is correct if it prefers the correct translation
        is_correct = all(correct_score > score for score in contrastive_scores)

        details = {
            "example_id": example.example_id,
            "pronoun_type": example.pronoun_type,
            "antecedent_distance": example.antecedent_distance,
            "correct_score": correct_score,
            "contrastive_scores": contrastive_scores,
            "is_correct": is_correct,
        }

        return is_correct, details

    def evaluate(
        self,
        examples: List[ContrastiveExample],
    ) -> ContrastiveResult:
        """
        Evaluate on a list of contrastive examples.

        Args:
            examples: List of ContrastiveExample objects

        Returns:
            ContrastiveResult with accuracy breakdown
        """
        result = ContrastiveResult()
        result.total_examples = len(examples)

        by_pronoun = defaultdict(lambda: {"correct": 0, "total": 0})
        by_distance = defaultdict(lambda: {"correct": 0, "total": 0})

        for example in examples:
            is_correct, details = self.evaluate_example(example)

            if is_correct:
                result.correct += 1

            result.examples_evaluated.append(details)

            # Track by pronoun type
            by_pronoun[example.pronoun_type]["total"] += 1
            if is_correct:
                by_pronoun[example.pronoun_type]["correct"] += 1

            # Track by distance
            by_distance[example.antecedent_distance]["total"] += 1
            if is_correct:
                by_distance[example.antecedent_distance]["correct"] += 1

        # Compute accuracies
        result.accuracy = result.correct / max(1, result.total_examples)

        for ptype, counts in by_pronoun.items():
            result.by_pronoun_type[ptype] = counts["correct"] / max(1, counts["total"])

        for dist, counts in by_distance.items():
            result.by_distance[dist] = counts["correct"] / max(1, counts["total"])

        return result


class ContraProDataset:
    """
    Loader for ContraPro-style evaluation data.

    ContraPro format: JSON with contrastive examples for pronoun translation.
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Args:
            data_path: Path to ContraPro JSON file
        """
        self.data_path = Path(data_path)
        self.examples = []

        if self.data_path.exists():
            self._load_data()

    def _load_data(self):
        """Load ContraPro data from JSON."""
        with open(self.data_path, "r") as f:
            data = json.load(f)

        for i, item in enumerate(data):
            example = ContrastiveExample(
                source_context=item.get("src_context", ""),
                source_current=item.get("src_current", item.get("src", "")),
                correct_translation=item.get("ref", item.get("correct", "")),
                contrastive_translations=item.get("contrastive", []),
                pronoun_type=item.get("pronoun_type", "unknown"),
                antecedent_distance=item.get("ante_distance", 0),
                antecedent=item.get("antecedent", ""),
                example_id=str(i),
            )
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


def create_synthetic_contrapro_examples(
    n_examples: int = 100,
) -> List[ContrastiveExample]:
    """
    Create synthetic contrastive examples for testing.

    This generates simple pronoun resolution examples
    where the pronoun should agree with an antecedent.
    """
    examples = []

    # Template: "The [noun] is [adjective]. [pronoun] is [property]."
    templates = [
        {
            "antecedent": "man",
            "pronoun_type": "he",
            "context": "The man is tall.",
            "current": "Er ist stark.",  # German: He is strong
            "correct": "He is strong.",
            "contrastive": ["She is strong.", "It is strong."],
        },
        {
            "antecedent": "woman",
            "pronoun_type": "she",
            "context": "The woman is intelligent.",
            "current": "Sie ist kreativ.",  # German: She is creative
            "correct": "She is creative.",
            "contrastive": ["He is creative.", "It is creative."],
        },
        {
            "antecedent": "book",
            "pronoun_type": "it",
            "context": "The book is interesting.",
            "current": "Es ist lang.",  # German: It is long
            "correct": "It is long.",
            "contrastive": ["He is long.", "She is long."],
        },
        {
            "antecedent": "children",
            "pronoun_type": "they",
            "context": "The children are playing.",
            "current": "Sie sind glücklich.",  # German: They are happy
            "correct": "They are happy.",
            "contrastive": ["He is happy.", "She is happy.", "It is happy."],
        },
    ]

    for i in range(n_examples):
        template = templates[i % len(templates)]
        distance = (i // len(templates)) % 5 + 1  # Vary distance

        example = ContrastiveExample(
            source_context=template["context"],
            source_current=template["current"],
            correct_translation=template["correct"],
            contrastive_translations=template["contrastive"],
            pronoun_type=template["pronoun_type"],
            antecedent_distance=distance,
            antecedent=template["antecedent"],
            example_id=f"synthetic_{i}",
        )
        examples.append(example)

    return examples


def evaluate_pronoun_accuracy(
    model,
    tokenizer,
    examples: List[ContrastiveExample],
    device: str = "cuda",
) -> ContrastiveResult:
    """
    Convenience function for pronoun accuracy evaluation.

    Args:
        model: NMT model
        tokenizer: Tokenizer
        examples: Contrastive examples
        device: Device

    Returns:
        ContrastiveResult
    """
    evaluator = ContrastivePronounEvaluator(model, tokenizer, device)
    return evaluator.evaluate(examples)
