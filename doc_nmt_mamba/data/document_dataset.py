"""
Document-Level Dataset for NMT.

Handles document boundaries and provides document-aware sampling.
"""

from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import random
import hashlib

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

from .tokenization import NMTTokenizer
from .augmentation import ConcatenationAugmenter, DocumentSample


def get_split_hash(text: str, seed: int = 42, val_ratio: float = 0.05, test_ratio: float = 0.05) -> str:
    """
    Determine split for a sample using deterministic hashing.

    This ensures stable splits even if the dataset is updated or reordered.
    Uses first 100 chars of text + seed to compute hash.

    Args:
        text: Sample text (uses first 100 chars)
        seed: Random seed for reproducibility
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set

    Returns:
        Split name: "train", "validation", or "test"
    """
    # Use first 100 chars to avoid hashing very long texts
    text_key = text[:100] if len(text) > 100 else text
    hash_input = f"{text_key}{seed}".encode('utf-8')
    h = int(hashlib.md5(hash_input).hexdigest(), 16)
    r = h / (2**128)  # Normalize to [0, 1)

    if r < test_ratio:
        return "test"
    elif r < test_ratio + val_ratio:
        return "validation"
    return "train"


class DocumentNMTDataset(Dataset):
    """
    Document-level NMT dataset.

    Features:
    - Document boundary preservation
    - CAT-N augmentation
    - Efficient tokenization with caching
    """

    def __init__(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        tokenizer: NMTTokenizer,
        doc_boundaries: Optional[List[int]] = None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 4096,
        max_tgt_length: int = 4096,
        cache_tokenization: bool = True,
    ):
        """
        Args:
            src_texts: List of source texts
            tgt_texts: List of target texts
            tokenizer: NMT tokenizer
            doc_boundaries: List of document start indices (optional)
            augmenter: Optional CAT-N augmenter
            max_src_length: Maximum source length
            max_tgt_length: Maximum target length
            cache_tokenization: Whether to cache tokenized results
        """
        assert len(src_texts) == len(tgt_texts), "Source and target must have same length"

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.doc_boundaries = doc_boundaries
        self.augmenter = augmenter
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.cache_tokenization = cache_tokenization

        # Prepare samples (with augmentation if enabled)
        self.samples = self._prepare_samples()

        # Tokenization cache
        self._token_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _prepare_samples(self) -> List[Tuple[str, str]]:
        """Prepare training samples, applying augmentation if available."""
        if self.augmenter is None:
            # No augmentation - return pairs directly
            return list(zip(self.src_texts, self.tgt_texts))

        # Group into documents and augment
        if self.doc_boundaries is not None:
            samples = []
            for i in range(len(self.doc_boundaries)):
                start = self.doc_boundaries[i]
                end = (
                    self.doc_boundaries[i + 1]
                    if i + 1 < len(self.doc_boundaries)
                    else len(self.src_texts)
                )
                doc = DocumentSample(
                    src_sentences=self.src_texts[start:end],
                    tgt_sentences=self.tgt_texts[start:end],
                    doc_id=str(i),
                )
                samples.extend(self.augmenter.augment_document(doc))
            return samples
        else:
            # Treat all as single document
            doc = DocumentSample(
                src_sentences=self.src_texts,
                tgt_sentences=self.tgt_texts,
            )
            return self.augmenter.augment_document(doc)

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for reproducible augmentation.

        Delegates to the augmenter if one exists. Should be called at
        the start of each epoch to ensure reproducibility across runs.

        Args:
            epoch: Current epoch number
        """
        if self.augmenter is not None and hasattr(self.augmenter, 'set_epoch'):
            self.augmenter.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        if self.cache_tokenization and idx in self._token_cache:
            src_ids, tgt_ids = self._token_cache[idx]
        else:
            src_text, tgt_text = self.samples[idx]
            src_ids, tgt_ids = self.tokenizer.encode_pair(
                src_text,
                tgt_text,
                max_src_length=self.max_src_length,
                max_tgt_length=self.max_tgt_length,
            )

            if self.cache_tokenization:
                self._token_cache[idx] = (src_ids, tgt_ids)

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_len": len(src_ids),
            "tgt_len": len(tgt_ids),
        }

    def refresh_augmentation(self):
        """Re-apply augmentation (for new epoch with different samples)."""
        if self.augmenter is not None:
            self.samples = self._prepare_samples()
            self._token_cache.clear()


class IWSLT14Dataset(DocumentNMTDataset):
    """
    IWSLT14 German-English dataset.

    Wrapper for HuggingFace datasets with proper preprocessing.
    Falls back to WMT14 if IWSLT is unavailable.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Optional[NMTTokenizer] = None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 4096,
        max_tgt_length: int = 4096,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            split: Dataset split ("train", "validation", "test")
            tokenizer: NMT tokenizer (creates default if None)
            augmenter: Optional CAT-N augmenter
            max_src_length: Maximum source length
            max_tgt_length: Maximum target length
            cache_dir: Cache directory for dataset
        """
        # Try multiple dataset sources
        src_texts = None
        tgt_texts = None

        # Option 1: Try IWSLT2017 with trust_remote_code
        try:
            dataset = load_dataset(
                "iwslt2017",
                "iwslt2017-de-en",
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            src_texts = [item["translation"]["de"] for item in dataset]
            tgt_texts = [item["translation"]["en"] for item in dataset]
        except Exception as e:
            print(f"IWSLT2017 loading failed: {e}")

        # Option 2: Fall back to WMT14 de-en
        if src_texts is None:
            try:
                # Map split names
                wmt_split = "train" if split == "train" else "validation" if split == "validation" else "test"
                dataset = load_dataset(
                    "wmt14",
                    "de-en",
                    split=wmt_split,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                src_texts = [item["translation"]["de"] for item in dataset]
                tgt_texts = [item["translation"]["en"] for item in dataset]
                print(f"Using WMT14 dataset instead ({len(src_texts)} samples)")
            except Exception as e:
                print(f"WMT14 loading failed: {e}")

        # Option 3: Use synthetic data for testing
        if src_texts is None:
            print("WARNING: Using synthetic data for testing. Replace with real data for training.")
            n_samples = 10000 if split == "train" else 1000
            src_texts = [f"Das ist ein Testsatz Nummer {i}." for i in range(n_samples)]
            tgt_texts = [f"This is test sentence number {i}." for i in range(n_samples)]

        # Create tokenizer if needed
        if tokenizer is None:
            tokenizer = NMTTokenizer(src_lang="de_DE", tgt_lang="en_XX")

        super().__init__(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            tokenizer=tokenizer,
            doc_boundaries=None,  # IWSLT doesn't have doc boundaries
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
        )


class OPUSBooksDataset(DocumentNMTDataset):
    """
    OPUS Books German-English dataset with DOCUMENT BOUNDARIES.

    CRITICAL FOR THESIS: This dataset preserves document structure because
    consecutive entries are from the same literary work (e.g., Jane Eyre).
    This enables proper document-level learning with CAT-N augmentation.

    Unlike shuffled sentence-level data (WMT14), this dataset maintains
    discourse coherence needed for pronoun resolution and entity tracking.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 4096,
        max_tgt_length: int = 4096,
        cache_dir: Optional[str] = None,
        val_ratio: float = 0.05,
    ):
        """
        Args:
            split: Dataset split ("train", "validation", "test")
            tokenizer: Tokenizer (CustomBPETokenizer or NMTTokenizer)
            augmenter: Optional CAT-N augmenter
            max_src_length: Maximum source length
            max_tgt_length: Maximum target length
            cache_dir: Cache directory for dataset
            val_ratio: Ratio for validation split (since OPUS Books only has train)
        """
        # Load OPUS Books dataset
        dataset = load_dataset("opus_books", "de-en", split="train", cache_dir=cache_dir)

        # Use hash-based splitting for stable train/val/test assignment
        # This ensures splits are consistent even if dataset order changes
        test_ratio = val_ratio  # Same ratio for test as validation

        # Extract texts using hash-based split (preserves order within each split)
        src_texts = []
        tgt_texts = []
        doc_boundaries = [0]  # Track document boundaries

        for i in range(len(dataset)):
            item = dataset[i]
            de_text = item["translation"]["de"]
            en_text = item["translation"]["en"]

            # Determine split using hash of text content (stable across dataset versions)
            sample_split = get_split_hash(
                en_text,  # Use target text for hashing
                seed=42,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

            # Only include samples belonging to requested split
            if sample_split != split:
                continue

            # Detect document boundaries from ID patterns
            # Simple heuristic: new document if we see metadata patterns
            is_metadata = (
                "Source:" in en_text or
                "Project Gutenberg" in en_text or
                len(en_text) < 20 and en_text.istitle()  # Title-like entries
            )

            if is_metadata and len(src_texts) > 0:
                doc_boundaries.append(len(src_texts))

            src_texts.append(de_text)
            tgt_texts.append(en_text)

        print(f"OPUS Books ({split}): {len(src_texts)} samples, {len(doc_boundaries)} documents")

        # Create tokenizer if needed
        if tokenizer is None:
            from .tokenization import create_tokenizer
            tokenizer = create_tokenizer(tokenizer_type="custom")

        super().__init__(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            tokenizer=tokenizer,
            doc_boundaries=doc_boundaries,  # CRITICAL: Document boundaries preserved
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
        )


class NewsCommentaryDataset(DocumentNMTDataset):
    """
    News Commentary dataset with document-level structure.

    News articles naturally form documents with discourse coherence.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 4096,
        max_tgt_length: int = 4096,
        cache_dir: Optional[str] = None,
        val_ratio: float = 0.05,
    ):
        """
        Args:
            split: Dataset split
            tokenizer: Tokenizer
            augmenter: Optional CAT-N augmenter
            max_src_length: Maximum source length
            max_tgt_length: Maximum target length
            cache_dir: Cache directory
            val_ratio: Validation split ratio
        """
        dataset = load_dataset("news_commentary", "de-en", split="train", cache_dir=cache_dir)

        # Use hash-based splitting for stable train/val/test assignment
        test_ratio = val_ratio

        src_texts = []
        tgt_texts = []

        for i in range(len(dataset)):
            item = dataset[i]
            de_text = item["translation"]["de"]
            en_text = item["translation"]["en"]

            # Determine split using hash of text content
            sample_split = get_split_hash(
                en_text,
                seed=42,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

            if sample_split == split:
                src_texts.append(de_text)
                tgt_texts.append(en_text)

        # News articles: treat every N sentences as a "document" for CAT-N
        # This approximates article boundaries
        doc_boundaries = list(range(0, len(src_texts), 10))

        print(f"News Commentary ({split}): {len(src_texts)} samples")

        if tokenizer is None:
            from .tokenization import create_tokenizer
            tokenizer = create_tokenizer(tokenizer_type="custom")

        super().__init__(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            tokenizer=tokenizer,
            doc_boundaries=doc_boundaries,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
        )


class StreamingDocumentDataset(IterableDataset):
    """
    Streaming dataset for large document collections.

    Memory-efficient: tokenizes on-the-fly.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: NMTTokenizer,
        augmenter: Optional[ConcatenationAugmenter] = None,
        max_src_length: int = 4096,
        max_tgt_length: int = 4096,
        shuffle_buffer: int = 10000,
    ):
        """
        Args:
            data_path: Path to data file (tsv or parallel files)
            tokenizer: NMT tokenizer
            augmenter: Optional augmenter
            max_src_length: Max source length
            max_tgt_length: Max target length
            shuffle_buffer: Buffer size for shuffling
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        """Iterate over samples."""
        buffer = []

        with open(self.data_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    src_text, tgt_text = parts[0], parts[1]

                    # Tokenize
                    src_ids, tgt_ids = self.tokenizer.encode_pair(
                        src_text,
                        tgt_text,
                        max_src_length=self.max_src_length,
                        max_tgt_length=self.max_tgt_length,
                    )

                    sample = {
                        "src_ids": src_ids,
                        "tgt_ids": tgt_ids,
                        "src_len": len(src_ids),
                        "tgt_len": len(tgt_ids),
                    }

                    buffer.append(sample)

                    if len(buffer) >= self.shuffle_buffer:
                        random.shuffle(buffer)
                        while len(buffer) > self.shuffle_buffer // 2:
                            yield buffer.pop()

        # Yield remaining
        random.shuffle(buffer)
        for sample in buffer:
            yield sample


def create_dataset(
    dataset_name: str = "opus_books",
    split: str = "train",
    tokenizer=None,
    augmenter: Optional[ConcatenationAugmenter] = None,
    **kwargs,
) -> DocumentNMTDataset:
    """
    Factory function to create datasets.

    Args:
        dataset_name: Dataset name
            - "opus_books": RECOMMENDED for document-level NMT (has document boundaries)
            - "news_commentary": News articles with document structure
            - "iwslt14": Legacy, falls back to WMT14 (shuffled, NOT recommended)
        split: Data split
        tokenizer: Optional tokenizer
        augmenter: Optional augmenter
        **kwargs: Additional dataset arguments

    Returns:
        Dataset instance

    CRITICAL FOR THESIS: Use "opus_books" or "news_commentary" for proper
    document-level learning. Do NOT use shuffled sentence-level data.
    """
    if dataset_name == "opus_books":
        return OPUSBooksDataset(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            **kwargs,
        )
    elif dataset_name == "news_commentary":
        return NewsCommentaryDataset(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            **kwargs,
        )
    elif dataset_name == "iwslt14":
        import warnings
        warnings.warn(
            "IWSLT14 falls back to WMT14 which is shuffled sentence-level data. "
            "This does NOT preserve document boundaries. "
            "Use dataset_name='opus_books' for document-level NMT."
        )
        return IWSLT14Dataset(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'opus_books' or 'news_commentary'.")
