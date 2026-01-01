"""
Data Augmentation for Document-Level NMT.

CAT-N Strategy:
- 50% probability: single sentence
- 50% probability: concatenate N consecutive sentences
- Improves generalization to longer sequences (COMET +3-5 points)
"""

import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DocumentSample:
    """A document sample with source and target sentences."""
    src_sentences: List[str]
    tgt_sentences: List[str]
    doc_id: Optional[str] = None


class ConcatenationAugmenter:
    """
    CAT-N Augmentation for document-level NMT.

    Strategy:
    - With probability p_concat: concatenate N consecutive sentences
    - Otherwise: return single sentence
    - Ensures source and target are aligned
    """

    def __init__(
        self,
        n_sentences: int = 5,
        p_concat: float = 0.5,
        separator: str = " ",
        min_concat: int = 2,
        max_concat: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_sentences: Maximum number of sentences to concatenate
            p_concat: Probability of concatenation (vs single sentence)
            separator: Separator between concatenated sentences
            min_concat: Minimum sentences to concatenate (when concatenating)
            max_concat: Maximum sentences (defaults to n_sentences)
            seed: Random seed for reproducibility
        """
        self.n_sentences = n_sentences
        self.p_concat = p_concat
        self.separator = separator
        self.min_concat = min_concat
        self.max_concat = max_concat or n_sentences

        if seed is not None:
            random.seed(seed)

    def augment_document(
        self,
        document: DocumentSample,
    ) -> List[Tuple[str, str]]:
        """
        Augment a document into training samples.

        Args:
            document: Document with parallel sentences

        Returns:
            List of (source, target) text pairs
        """
        samples = []
        src_sents = document.src_sentences
        tgt_sents = document.tgt_sentences

        assert len(src_sents) == len(tgt_sents), "Source and target must have same length"

        i = 0
        while i < len(src_sents):
            if random.random() < self.p_concat and len(src_sents) - i >= self.min_concat:
                # Concatenate multiple sentences
                n = random.randint(self.min_concat, min(self.max_concat, len(src_sents) - i))
                src_text = self.separator.join(src_sents[i : i + n])
                tgt_text = self.separator.join(tgt_sents[i : i + n])
                samples.append((src_text, tgt_text))
                i += n
            else:
                # Single sentence
                samples.append((src_sents[i], tgt_sents[i]))
                i += 1

        return samples

    def augment_batch(
        self,
        documents: List[DocumentSample],
    ) -> List[Tuple[str, str]]:
        """
        Augment multiple documents.

        Args:
            documents: List of documents

        Returns:
            Combined list of (source, target) pairs
        """
        all_samples = []
        for doc in documents:
            all_samples.extend(self.augment_document(doc))
        return all_samples


class RandomConcatAugmenter:
    """
    Random concatenation augmenter.

    More flexible: randomly sample sentence count each time.
    """

    def __init__(
        self,
        min_sentences: int = 1,
        max_sentences: int = 10,
        separator: str = " ",
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            min_sentences: Minimum sentences per sample
            max_sentences: Maximum sentences per sample
            separator: Separator between sentences
            weights: Optional weights for each sentence count (1 to max)
        """
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.separator = separator

        if weights is None:
            # Default: exponentially decreasing probability for longer sequences
            self.weights = [1.0 / (i ** 0.5) for i in range(1, max_sentences + 1)]
        else:
            self.weights = weights

        # Normalize weights for the valid range
        valid_weights = self.weights[min_sentences - 1 : max_sentences]
        total = sum(valid_weights)
        self.normalized_weights = [w / total for w in valid_weights]

    def sample_length(self) -> int:
        """Sample number of sentences to concatenate."""
        lengths = list(range(self.min_sentences, self.max_sentences + 1))
        return random.choices(lengths, weights=self.normalized_weights)[0]

    def augment_document(
        self,
        document: DocumentSample,
    ) -> List[Tuple[str, str]]:
        """Augment document with random concatenation lengths."""
        samples = []
        src_sents = document.src_sentences
        tgt_sents = document.tgt_sentences

        i = 0
        while i < len(src_sents):
            n = min(self.sample_length(), len(src_sents) - i)
            src_text = self.separator.join(src_sents[i : i + n])
            tgt_text = self.separator.join(tgt_sents[i : i + n])
            samples.append((src_text, tgt_text))
            i += n

        return samples


def create_augmenter(
    strategy: str = "cat_n",
    **kwargs,
) -> ConcatenationAugmenter:
    """
    Factory function for augmenters.

    Args:
        strategy: "cat_n" or "random"
        **kwargs: Augmenter-specific arguments

    Returns:
        Augmenter instance
    """
    if strategy == "cat_n":
        return ConcatenationAugmenter(**kwargs)
    elif strategy == "random":
        return RandomConcatAugmenter(**kwargs)
    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")
