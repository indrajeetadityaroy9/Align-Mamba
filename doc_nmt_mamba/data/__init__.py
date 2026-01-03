"""
Data pipeline for Document-Level NMT.

Consolidated structure:
- tokenization.py: CustomBPETokenizer, NMTTokenizer
- dataset.py: All datasets (NMT + MQAR)
- collator.py: Collation + Augmentation

Provides:
- CustomBPETokenizer: 32K vocab (RECOMMENDED)
- NMTTokenizer: mBART-based tokenization (250K vocab)
- ConcatenationAugmenter: CAT-N strategy for document-level learning
- DocumentNMTDataset: Dataset with document awareness
- PackedSequenceCollator: Critical for H100 efficiency (20-30% speedup)
- MQARDataset: Synthetic MQAR task for state capacity testing
"""

from .tokenization import CustomBPETokenizer, NMTTokenizer, create_tokenizer

from .collator import (
    # Data classes
    DocumentSample,
    # Collators
    PackedSequenceCollator,
    PaddedSequenceCollator,
    DynamicBatchCollator,
    LabelShiftCollator,
    MQARCollator,
    create_collator,
    # Augmenters
    ConcatenationAugmenter,
    RandomConcatAugmenter,
    create_augmenter,
)

from .dataset import (
    # NMT Datasets
    DocumentNMTDataset,
    IWSLT14Dataset,
    OPUSBooksDataset,
    NewsCommentaryDataset,
    StreamingDocumentDataset,
    create_dataset,
    # MQAR Synthetic
    MQARConfig,
    MQARDataset,
    MQARCurriculumGenerator,
    compute_mqar_accuracy,
)

__all__ = [
    # Tokenization
    "CustomBPETokenizer",
    "NMTTokenizer",
    "create_tokenizer",
    # Data classes
    "DocumentSample",
    # Collators
    "PackedSequenceCollator",
    "PaddedSequenceCollator",
    "DynamicBatchCollator",
    "LabelShiftCollator",
    "MQARCollator",
    "create_collator",
    # Augmenters
    "ConcatenationAugmenter",
    "RandomConcatAugmenter",
    "create_augmenter",
    # NMT Datasets
    "DocumentNMTDataset",
    "IWSLT14Dataset",
    "OPUSBooksDataset",
    "NewsCommentaryDataset",
    "StreamingDocumentDataset",
    "create_dataset",
    # MQAR Synthetic
    "MQARConfig",
    "MQARDataset",
    "MQARCurriculumGenerator",
    "compute_mqar_accuracy",
]
