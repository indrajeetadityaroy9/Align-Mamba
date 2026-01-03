"""
Analysis Modules for Document-Level NMT Evaluation.

This file consolidates specialized evaluation tools:
- Alignment: SubwordToWordMapper, AlignmentEvaluator (AER metric)
- ContraPro: ContrastivePronounEvaluator (pronoun disambiguation)
- Entity: EntityRecallAnalyzer (named entity preservation)
- Length: LengthSensitivityAnalyzer, ExtrapolationTester

Key metrics for document-level NMT:
- AER < 0.30: Good alignment quality
- ContraPro +10pp vs sentence-level: Document context utilization
- Entity Recall > 90%: Consistent entity translation
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import json
import time
import re
import warnings

import torch
import torch.nn.functional as F

# Optional spaCy for NER
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# =============================================================================
# Alignment Evaluation (AER)
# =============================================================================

@dataclass
class AlignmentResult:
    """Results from alignment evaluation."""
    aer: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    total_examples: int = 0
    per_example_aer: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"AlignmentResult(AER={self.aer:.4f}, P={self.precision:.4f}, R={self.recall:.4f})"


class SubwordToWordMapper:
    """
    Maps BPE token indices back to word indices for AER computation.

    CRITICAL: awesome-align outputs word-level alignments. Our model uses 32k BPE.
    Without mapping, you get shape mismatch (Gold: Word 3→5, Model: Token 7→12).
    """

    def __init__(self, tokenizer=None, word_boundary_prefix: str = "▁"):
        self.tokenizer = tokenizer
        self.word_boundary_prefix = word_boundary_prefix

    def build_token_to_word_map(self, tokens: List[str]) -> List[int]:
        """Build mapping from token indices to word indices."""
        token_to_word = []
        current_word = -1

        for token in tokens:
            if token.startswith(self.word_boundary_prefix) or current_word == -1:
                current_word += 1
            token_to_word.append(current_word)

        return token_to_word

    def build_map_from_text(
        self,
        text: str,
        encoded_ids: Optional[List[int]] = None,
    ) -> Tuple[List[int], int]:
        """Build token-to-word mapping from original text."""
        words = text.split()
        n_words = len(words)

        if self.tokenizer is None:
            warnings.warn("No tokenizer provided. Assuming 1:1 word-token mapping.")
            return list(range(n_words)), n_words

        if encoded_ids is None:
            encoded_ids = self.tokenizer.encode(text)

        tokens = self.tokenizer.convert_ids_to_tokens(encoded_ids)
        token_to_word = self.build_token_to_word_map(tokens)

        return token_to_word, n_words

    def aggregate_attention_to_words(
        self,
        attn_weights: torch.Tensor,
        src_token_to_word: List[int],
        tgt_token_to_word: List[int],
        n_src_words: int,
        n_tgt_words: int,
        aggregation: str = "max",
    ) -> torch.Tensor:
        """Aggregate token-level attention to word-level attention."""
        T_tgt, T_src = attn_weights.shape
        word_attn = torch.zeros(n_tgt_words, n_src_words, device=attn_weights.device)

        if aggregation == "max":
            for t_idx in range(T_tgt):
                for s_idx in range(T_src):
                    t_word = tgt_token_to_word[t_idx]
                    s_word = src_token_to_word[s_idx]
                    word_attn[t_word, s_word] = max(
                        word_attn[t_word, s_word].item(),
                        attn_weights[t_idx, s_idx].item()
                    )
        elif aggregation == "mean":
            word_count = torch.zeros(n_tgt_words, n_src_words, device=attn_weights.device)
            for t_idx in range(T_tgt):
                for s_idx in range(T_src):
                    t_word = tgt_token_to_word[t_idx]
                    s_word = src_token_to_word[s_idx]
                    word_attn[t_word, s_word] += attn_weights[t_idx, s_idx]
                    word_count[t_word, s_word] += 1
            word_attn = word_attn / (word_count + 1e-8)
        else:
            for t_idx in range(T_tgt):
                for s_idx in range(T_src):
                    t_word = tgt_token_to_word[t_idx]
                    s_word = src_token_to_word[s_idx]
                    word_attn[t_word, s_word] += attn_weights[t_idx, s_idx]

        return word_attn


def load_awesome_align_alignments(file_path: str) -> List[Set[Tuple[int, int]]]:
    """Load alignments from awesome-align output file."""
    alignments = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                alignments.append(set())
                continue
            pairs = set()
            for pair in line.split():
                if "-" in pair:
                    src_idx, tgt_idx = pair.split("-")
                    pairs.add((int(src_idx), int(tgt_idx)))
            alignments.append(pairs)
    return alignments


class AlignmentEvaluator:
    """Evaluator for alignment quality using AER metric."""

    def __init__(self, tokenizer=None, cross_attention_layers: Optional[List[int]] = None):
        self.tokenizer = tokenizer
        self.cross_attention_layers = cross_attention_layers
        self.mapper = SubwordToWordMapper(tokenizer)

    def compute_aer(
        self,
        predicted: Set[Tuple[int, int]],
        gold_sure: Set[Tuple[int, int]],
        gold_possible: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[float, float, float]:
        """Compute Alignment Error Rate."""
        if gold_possible is None:
            gold_possible = gold_sure

        a_intersect_s = len(predicted & gold_sure)
        a_intersect_p = len(predicted & gold_possible)

        denominator = len(predicted) + len(gold_sure)
        if denominator == 0:
            return 0.0, 0.0, 0.0

        aer = 1 - (a_intersect_s + a_intersect_p) / denominator
        precision = a_intersect_p / len(predicted) if predicted else 0.0
        recall = a_intersect_s / len(gold_sure) if gold_sure else 0.0

        return aer, precision, recall


# =============================================================================
# Contrastive Pronoun Evaluation (ContraPro)
# =============================================================================

@dataclass
class ContrastiveExample:
    """A contrastive evaluation example."""
    source_context: str
    source_current: str
    correct_translation: str
    contrastive_translations: List[str]
    pronoun_type: str
    antecedent_distance: int
    antecedent: str
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
        return f"ContrastiveResult(accuracy={self.accuracy:.2%}, correct={self.correct}/{self.total_examples})"


class ContrastivePronounEvaluator:
    """Contrastive evaluation for anaphoric pronouns."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def score_translation(self, source: str, target: str) -> float:
        """Compute log probability of target given source."""
        src_ids, tgt_ids = self.tokenizer.encode_pair(source, target)
        src_ids = src_ids.unsqueeze(0).to(self.device)
        tgt_ids = tgt_ids.unsqueeze(0).to(self.device)

        self.model.eval()
        logits = self.model(src_ids, tgt_ids[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)

        target_tokens = tgt_ids[:, 1:]
        token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.mean().item()

    def evaluate_example(self, example: ContrastiveExample) -> Tuple[bool, Dict]:
        """Evaluate a single contrastive example."""
        full_source = f"{example.source_context} {example.source_current}"
        correct_score = self.score_translation(full_source, example.correct_translation)
        contrastive_scores = [
            self.score_translation(full_source, trans)
            for trans in example.contrastive_translations
        ]

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

    def evaluate(self, examples: List[ContrastiveExample]) -> ContrastiveResult:
        """Evaluate on a list of contrastive examples."""
        result = ContrastiveResult()
        result.total_examples = len(examples)

        by_pronoun = defaultdict(lambda: {"correct": 0, "total": 0})
        by_distance = defaultdict(lambda: {"correct": 0, "total": 0})

        for example in examples:
            is_correct, details = self.evaluate_example(example)
            if is_correct:
                result.correct += 1
            result.examples_evaluated.append(details)

            by_pronoun[example.pronoun_type]["total"] += 1
            if is_correct:
                by_pronoun[example.pronoun_type]["correct"] += 1

            by_distance[example.antecedent_distance]["total"] += 1
            if is_correct:
                by_distance[example.antecedent_distance]["correct"] += 1

        result.accuracy = result.correct / max(1, result.total_examples)

        for ptype, counts in by_pronoun.items():
            result.by_pronoun_type[ptype] = counts["correct"] / max(1, counts["total"])
        for dist, counts in by_distance.items():
            result.by_distance[dist] = counts["correct"] / max(1, counts["total"])

        return result


class ContraProDataset:
    """Loader for ContraPro-style evaluation data."""

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.examples = []
        if self.data_path.exists():
            self._load_data()

    def _load_data(self):
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


# =============================================================================
# Named Entity Recall Analysis
# =============================================================================

@dataclass
class EntityRecallResult:
    """Results from entity recall analysis."""
    overall_recall: float = 0.0
    overall_precision: float = 0.0
    overall_f1: float = 0.0
    total_source_entities: int = 0
    total_found_in_translation: int = 0
    by_entity_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_frequency: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missing_entities: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"EntityRecallResult(recall={self.overall_recall:.2%}, precision={self.overall_precision:.2%}, F1={self.overall_f1:.2%})"


class SimpleNER:
    """Simple rule-based named entity recognizer (fallback when spaCy unavailable)."""

    TITLE_PATTERN = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+")
    CAPITALIZED_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
    STOPWORDS = {
        "The", "A", "An", "This", "That", "These", "Those",
        "I", "He", "She", "It", "We", "They", "You",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    }

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract potential named entities from text."""
        entities = []

        for match in self.TITLE_PATTERN.finditer(text):
            entities.append((match.group(), "PERSON"))

        sentences = text.split(". ")
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if i == 0:
                    continue
                if word in self.STOPWORDS:
                    continue
                if word[0].isupper() and word not in [e[0] for e in entities]:
                    if any(c.isdigit() for c in word):
                        entities.append((word, "NUMBER"))
                    else:
                        entities.append((word, "ENTITY"))

        return entities


class SpaCyNER:
    """SpaCy-based named entity recognizer."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy not available. Install with: pip install spacy")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities using spaCy."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]


class EntityRecallAnalyzer:
    """Analyzes named entity recall in translations."""

    def __init__(self, src_lang: str = "de", tgt_lang: str = "en", use_spacy: bool = True):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if use_spacy and SPACY_AVAILABLE:
            try:
                self.src_ner = SpaCyNER(f"{src_lang}_core_news_sm")
                self.tgt_ner = SpaCyNER(f"{tgt_lang}_core_web_sm")
                self.use_spacy = True
            except Exception as e:
                warnings.warn(f"Failed to load spaCy models: {e}. Using simple NER.")
                self.src_ner = SimpleNER()
                self.tgt_ner = SimpleNER()
                self.use_spacy = False
        else:
            self.src_ner = SimpleNER()
            self.tgt_ner = SimpleNER()
            self.use_spacy = False

    def normalize_entity(self, entity: str) -> str:
        return entity.lower().strip()

    def entity_match(self, src_entity: str, tgt_entities: Set[str]) -> bool:
        src_norm = self.normalize_entity(src_entity)
        if src_norm in tgt_entities:
            return True
        for tgt in tgt_entities:
            if src_norm in tgt or tgt in src_norm:
                return True
        return False

    def analyze(self, sources: List[str], translations: List[str]) -> EntityRecallResult:
        """Analyze entity recall across a corpus."""
        result = EntityRecallResult()
        all_missing = []
        by_type = defaultdict(lambda: {"found": 0, "total": 0})
        entity_counts = defaultdict(int)

        for src, tgt in zip(sources, translations):
            src_entities = self.src_ner.extract_entities(src)
            tgt_entities = self.tgt_ner.extract_entities(tgt)

            tgt_entity_texts = {self.normalize_entity(e[0]) for e in tgt_entities}
            result.total_source_entities += len(src_entities)

            found = 0
            for ent_text, ent_type in src_entities:
                entity_counts[ent_text] += 1
                by_type[ent_type]["total"] += 1

                if self.entity_match(ent_text, tgt_entity_texts):
                    found += 1
                    by_type[ent_type]["found"] += 1
                else:
                    all_missing.append(ent_text)

            result.total_found_in_translation += found

        result.overall_recall = result.total_found_in_translation / max(1, result.total_source_entities)

        for ent_type, counts in by_type.items():
            result.by_entity_type[ent_type] = {
                "recall": counts["found"] / max(1, counts["total"]),
                "total": counts["total"],
            }

        missing_counts = Counter(all_missing)
        result.missing_entities = [ent for ent, _ in missing_counts.most_common(20)]

        return result


# =============================================================================
# Length Sensitivity Analysis
# =============================================================================

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
        lines = ["Length Sensitivity Analysis", "=" * 40]
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
    """Analyzes model performance across different sequence lengths."""

    def __init__(self, model, tokenizer, device: str = "cuda", max_length: int = 8192):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def measure_memory(self, batch_size: int = 1, seq_length: int = 1024) -> float:
        """Measure GPU memory consumption for given sequence length."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        src_ids = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)
        tgt_ids = torch.randint(0, 1000, (batch_size, seq_length // 2), device=self.device)

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
        """Measure inference speed for given sequence length."""
        src_ids = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)
        tgt_ids = torch.randint(0, 1000, (batch_size, seq_length // 2), device=self.device)

        self.model.eval()

        for _ in range(n_warmup):
            _ = self.model(src_ids, tgt_ids)
            torch.cuda.synchronize()

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
        """Analyze memory and speed scaling."""
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


class ExtrapolationTester:
    """Tests model's ability to extrapolate to longer sequences."""

    def __init__(self, model, tokenizer, training_max_length: int = 2048, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.training_max_length = training_max_length
        self.device = device

    @torch.no_grad()
    def test_extrapolation(self, test_lengths: List[int] = None) -> Dict[int, Dict]:
        """Test model on lengths beyond training."""
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
                src_ids = torch.randint(0, 1000, (1, length), device=self.device)

                start = time.perf_counter()
                encoder_out = self.model.encode(src_ids)
                encode_time = time.perf_counter() - start

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


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_pronoun_accuracy(
    model,
    tokenizer,
    examples: List[ContrastiveExample],
    device: str = "cuda",
) -> ContrastiveResult:
    """Convenience function for pronoun accuracy evaluation."""
    evaluator = ContrastivePronounEvaluator(model, tokenizer, device)
    return evaluator.evaluate(examples)


def analyze_entity_recall(
    sources: List[str],
    translations: List[str],
    src_lang: str = "de",
    tgt_lang: str = "en",
) -> EntityRecallResult:
    """Convenience function for entity recall analysis."""
    analyzer = EntityRecallAnalyzer(src_lang, tgt_lang)
    return analyzer.analyze(sources, translations)


def analyze_length_sensitivity(
    model,
    tokenizer,
    device: str = "cuda",
) -> LengthAnalysisResult:
    """Convenience function for length sensitivity analysis."""
    analyzer = LengthSensitivityAnalyzer(model, tokenizer, device)
    memory_scaling, speed_scaling = analyzer.analyze_scaling()

    result = LengthAnalysisResult()
    result.memory_scaling = memory_scaling
    result.speed_scaling = speed_scaling

    for length in memory_scaling:
        result.by_length[str(length)] = {
            "memory_gb": memory_scaling.get(length, 0),
            "tokens_per_sec": speed_scaling.get(length, 0),
        }

    return result
