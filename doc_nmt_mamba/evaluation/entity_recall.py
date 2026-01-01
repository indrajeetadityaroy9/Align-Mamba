"""
Named Entity Recall Analysis for Document-Level NMT.

Evaluates how well the model preserves named entities from source to translation.
Important for document coherence - entities should be consistently translated.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re
import warnings

# Optional spaCy for NER
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("spaCy not available. Using simple NER fallback.")


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
        return (
            f"EntityRecallResult(recall={self.overall_recall:.2%}, "
            f"precision={self.overall_precision:.2%}, "
            f"F1={self.overall_f1:.2%})"
        )


class SimpleNER:
    """
    Simple rule-based named entity recognizer.

    Fallback when spaCy is not available.
    Uses capitalization patterns to detect potential entities.
    """

    # Common entity patterns
    TITLE_PATTERN = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+")
    CAPITALIZED_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

    # Common non-entity capitalized words
    STOPWORDS = {
        "The", "A", "An", "This", "That", "These", "Those",
        "I", "He", "She", "It", "We", "They", "You",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    }

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract potential named entities from text.

        Args:
            text: Input text

        Returns:
            List of (entity_text, entity_type) tuples
        """
        entities = []

        # Find titled names
        for match in self.TITLE_PATTERN.finditer(text):
            entities.append((match.group(), "PERSON"))

        # Find other capitalized sequences (at non-sentence starts)
        sentences = text.split(". ")
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if i == 0:
                    continue  # Skip sentence-initial words

                if word in self.STOPWORDS:
                    continue

                if word[0].isupper() and word not in [e[0] for e in entities]:
                    # Classify based on patterns
                    if any(c.isdigit() for c in word):
                        entities.append((word, "NUMBER"))
                    else:
                        entities.append((word, "ENTITY"))

        return entities


class SpaCyNER:
    """
    SpaCy-based named entity recognizer.

    More accurate than simple NER.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Args:
            model_name: spaCy model name
        """
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy not available. Install with: pip install spacy")

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Download if not available
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities using spaCy.

        Args:
            text: Input text

        Returns:
            List of (entity_text, entity_type) tuples
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]


class EntityRecallAnalyzer:
    """
    Analyzes named entity recall in translations.

    Measures how well entities from the source appear in the translation.
    """

    def __init__(
        self,
        src_lang: str = "de",
        tgt_lang: str = "en",
        use_spacy: bool = True,
    ):
        """
        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            use_spacy: Whether to use spaCy (if available)
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Initialize NER
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
        """Normalize entity for matching."""
        return entity.lower().strip()

    def entity_match(
        self,
        src_entity: str,
        tgt_entities: Set[str],
    ) -> bool:
        """
        Check if source entity appears in target entities.

        Uses fuzzy matching for translations.
        """
        src_norm = self.normalize_entity(src_entity)

        # Exact match
        if src_norm in tgt_entities:
            return True

        # Substring match (for partial translations)
        for tgt in tgt_entities:
            if src_norm in tgt or tgt in src_norm:
                return True

        return False

    def analyze_pair(
        self,
        source: str,
        translation: str,
    ) -> Dict:
        """
        Analyze entity recall for a single source-translation pair.

        Args:
            source: Source text
            translation: Translation text

        Returns:
            Dictionary with recall metrics
        """
        # Extract entities
        src_entities = self.src_ner.extract_entities(source)
        tgt_entities = self.tgt_ner.extract_entities(translation)

        src_entity_texts = {self.normalize_entity(e[0]) for e in src_entities}
        tgt_entity_texts = {self.normalize_entity(e[0]) for e in tgt_entities}

        # Count matches
        found = 0
        missing = []
        for src_ent in src_entities:
            if self.entity_match(src_ent[0], tgt_entity_texts):
                found += 1
            else:
                missing.append(src_ent[0])

        recall = found / max(1, len(src_entities))
        precision = found / max(1, len(tgt_entities)) if tgt_entities else 0.0

        return {
            "source_entities": src_entities,
            "translation_entities": tgt_entities,
            "found": found,
            "missing": missing,
            "recall": recall,
            "precision": precision,
        }

    def analyze(
        self,
        sources: List[str],
        translations: List[str],
    ) -> EntityRecallResult:
        """
        Analyze entity recall across a corpus.

        Args:
            sources: Source texts
            translations: Translation texts

        Returns:
            EntityRecallResult with aggregated metrics
        """
        result = EntityRecallResult()

        all_missing = []
        by_type = defaultdict(lambda: {"found": 0, "total": 0})
        entity_counts = defaultdict(int)

        for src, tgt in zip(sources, translations):
            pair_result = self.analyze_pair(src, tgt)

            result.total_source_entities += len(pair_result["source_entities"])
            result.total_found_in_translation += pair_result["found"]
            all_missing.extend(pair_result["missing"])

            # Track by entity type
            for ent_text, ent_type in pair_result["source_entities"]:
                by_type[ent_type]["total"] += 1
                entity_counts[ent_text] += 1

                if self.entity_match(ent_text, {self.normalize_entity(e[0]) for e in pair_result["translation_entities"]}):
                    by_type[ent_type]["found"] += 1

        # Compute overall metrics
        result.overall_recall = (
            result.total_found_in_translation / max(1, result.total_source_entities)
        )

        # By-type metrics
        for ent_type, counts in by_type.items():
            result.by_entity_type[ent_type] = {
                "recall": counts["found"] / max(1, counts["total"]),
                "total": counts["total"],
            }

        # By-frequency metrics
        freq_buckets = {"rare": (1, 2), "medium": (3, 10), "frequent": (11, float("inf"))}
        for bucket_name, (min_freq, max_freq) in freq_buckets.items():
            bucket_found = 0
            bucket_total = 0

            for ent, count in entity_counts.items():
                if min_freq <= count <= max_freq:
                    bucket_total += 1
                    if ent not in all_missing:
                        bucket_found += 1

            if bucket_total > 0:
                result.by_frequency[bucket_name] = {
                    "recall": bucket_found / bucket_total,
                    "total": bucket_total,
                }

        # Store most common missing entities
        from collections import Counter
        missing_counts = Counter(all_missing)
        result.missing_entities = [ent for ent, _ in missing_counts.most_common(20)]

        return result


def analyze_entity_recall(
    sources: List[str],
    translations: List[str],
    src_lang: str = "de",
    tgt_lang: str = "en",
) -> EntityRecallResult:
    """
    Convenience function for entity recall analysis.

    Args:
        sources: Source texts
        translations: Translation texts
        src_lang: Source language
        tgt_lang: Target language

    Returns:
        EntityRecallResult
    """
    analyzer = EntityRecallAnalyzer(src_lang, tgt_lang)
    return analyzer.analyze(sources, translations)
