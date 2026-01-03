"""
Data pipeline for Document-Level NMT.

Provides:
- CustomBPETokenizer: 32K vocab (RECOMMENDED for thesis)
- NMTTokenizer: mBART-based tokenization (250K vocab, NOT recommended)
- ConcatenationAugmenter: CAT-N strategy for document-level learning
- DocumentNMTDataset: Dataset with document awareness
- PackedSequenceCollator: Critical for H100 efficiency
- MQARDataset: Synthetic MQAR task for state capacity testing
"""

from .tokenization import CustomBPETokenizer, NMTTokenizer, create_tokenizer
from .augmentation import (
    DocumentSample,
    ConcatenationAugmenter,
    RandomConcatAugmenter,
    create_augmenter,
)
from .document_dataset import (
    DocumentNMTDataset,
    IWSLT14Dataset,
    OPUSBooksDataset,
    NewsCommentaryDataset,
    StreamingDocumentDataset,
    create_dataset,
)
from .collation import (
    PackedSequenceCollator,
    PaddedSequenceCollator,
    DynamicBatchCollator,
    LabelShiftCollator,
    create_collator,
)
from .synthetic import (
    MQARConfig,
    MQARDataset,
    MQARCurriculumGenerator,
    MQARCollator,
    compute_mqar_accuracy,
    create_mqar_decoder_only_format,
)

__all__ = [
    # Tokenization
    "CustomBPETokenizer",
    "NMTTokenizer",
    "create_tokenizer",
    # Augmentation
    "DocumentSample",
    "ConcatenationAugmenter",
    "RandomConcatAugmenter",
    "create_augmenter",
    # Datasets
    "DocumentNMTDataset",
    "IWSLT14Dataset",
    "OPUSBooksDataset",
    "NewsCommentaryDataset",
    "StreamingDocumentDataset",
    "create_dataset",
    # Collation
    "PackedSequenceCollator",
    "PaddedSequenceCollator",
    "DynamicBatchCollator",
    "LabelShiftCollator",
    "create_collator",
    # Synthetic (MQAR)
    "MQARConfig",
    "MQARDataset",
    "MQARCurriculumGenerator",
    "MQARCollator",
    "compute_mqar_accuracy",
    "create_mqar_decoder_only_format",
]
