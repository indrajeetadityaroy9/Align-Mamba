"""Datasets for State Capacity experiments.

This module provides:
- MQARDataset: Synthetic Multi-Query Associative Recall for capacity testing
- OPUSBooksDataset: Real document-level translation for application experiments
- DocumentNMTDataset: Base class for NMT datasets
"""

from typing import List, Dict, Optional, Union, Any
import hashlib
import logging

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from .collator import DocumentSample, DocumentConcatenationAugmenter

from .mqar import (
    MQARConfig,
    MQARDataset,
    MQARCurriculumGenerator,
    compute_mqar_accuracy,
)

logger = logging.getLogger(__name__)


def get_split_hash(
    text: str,
    num_buckets: int = 100,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> str:
    """Deterministic hash-based split - consistent across runs without storing assignments."""
    hash_input = f"42:{text}"
    hash_bytes = hashlib.md5(hash_input.encode('utf-8')).digest()
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    bucket = hash_int % num_buckets

    test_threshold = int(test_ratio * num_buckets)
    val_threshold = test_threshold + int(val_ratio * num_buckets)

    if bucket < test_threshold:
        return "test"
    elif bucket < val_threshold:
        return "validation"
    else:
        return "train"


class DocumentNMTDataset(Dataset):
    """Base class for document-level NMT. Use directly with src/tgt_texts or subclass with _load_data()."""

    def __init__(
        self,
        split: str = "train",
        tokenizer: Any = None,
        augmenter: Optional[DocumentConcatenationAugmenter] = None,
        max_src_length: int = 512,
        max_tgt_length: int = 512,
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self._epoch = 0

        if src_texts is not None and tgt_texts is not None:
            self._samples = [
                DocumentSample(
                    src_sentences=[src],
                    tgt_sentences=[tgt],
                    doc_id=f"doc_{i}",
                )
                for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts))
            ]
        else:
            self._samples = self._load_data()

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        if self.augmenter:
            self.augmenter.set_epoch(epoch)

    def _load_data(self) -> List[DocumentSample]:
        return []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.augmenter is not None and self.split == "train":
            src_text, tgt_text = self.augmenter(self._samples, idx)
        else:
            sample = self._samples[idx]
            if len(sample) > 0:
                src_text, tgt_text = sample.get_sentence_pair(0)
            else:
                src_text, tgt_text = "", ""

        if self.tokenizer is not None:
            src_ids, tgt_ids = self.tokenizer.encode_pair(
                src_text, tgt_text,
                max_src_length=self.max_src_length,
                max_tgt_length=self.max_tgt_length,
            )
        else:
            return {'src_text': src_text, 'tgt_text': tgt_text}

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
        }


class OPUSBooksDataset(DocumentNMTDataset):
    """OPUS Books with document structure - groups sentences into pseudo-documents.

    Used for application experiments (document-level translation) to show
    that the State Capacity mechanism transfers to real translation tasks.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Any = None,
        augmenter: Optional[DocumentConcatenationAugmenter] = None,
        max_src_length: int = 512,
        max_tgt_length: int = 512,
        src_lang: str = "de",
        tgt_lang: str = "en",
        sentences_per_doc: int = 20,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sentences_per_doc = sentences_per_doc
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
        )

    def _load_data(self) -> List[DocumentSample]:
        dataset = load_dataset(
            "opus_books",
            f"{self.src_lang}-{self.tgt_lang}",
            split="train",
            trust_remote_code=True,
        )

        all_samples = []
        for item in dataset:
            translation = item['translation']
            all_samples.append((
                translation[self.src_lang],
                translation[self.tgt_lang],
            ))

        documents = []
        current_src = []
        current_tgt = []

        for src, tgt in all_samples:
            current_src.append(src)
            current_tgt.append(tgt)

            if len(current_src) >= self.sentences_per_doc:
                documents.append(DocumentSample(
                    src_sentences=current_src.copy(),
                    tgt_sentences=current_tgt.copy(),
                ))
                current_src = []
                current_tgt = []

        if current_src:
            documents.append(DocumentSample(
                src_sentences=current_src,
                tgt_sentences=current_tgt,
            ))

        train_docs = []
        val_docs = []
        test_docs = []

        for i, doc in enumerate(documents):
            bucket = get_split_hash(doc.src_sentences[0], 100)
            if bucket < 80:
                train_docs.append(doc)
            elif bucket < 90:
                val_docs.append(doc)
            else:
                test_docs.append(doc)

        if self.split == "train":
            return train_docs
        elif self.split == "validation":
            return val_docs
        else:
            return test_docs


def create_dataset(
    dataset_name: str = "mqar",
    split: str = "train",
    tokenizer: Any = None,
    augmenter: Optional[DocumentConcatenationAugmenter] = None,
    max_src_length: int = 512,
    max_tgt_length: int = 512,
    **kwargs,
) -> Union[DocumentNMTDataset, MQARDataset]:
    """
    Factory for creating datasets.

    Available datasets:
    - 'mqar': Synthetic MQAR for state capacity testing (default)
    - 'opus_books': OPUS Books with document structure for application experiments
    """
    if dataset_name == "opus_books":
        return OPUSBooksDataset(
            split=split,
            tokenizer=tokenizer,
            augmenter=augmenter,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
            **kwargs,
        )
    elif dataset_name == "mqar":
        config = kwargs.get('config', MQARConfig())
        num_samples = kwargs.get('num_samples', 10000)
        return MQARDataset(
            config=config,
            num_samples=num_samples,
            split=split,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: mqar, opus_books")
