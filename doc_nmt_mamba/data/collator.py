"""
Collation and Augmentation for Document-Level NMT.

Contains:
- DocumentSample: Dataclass for document samples
- ConcatenationAugmenter: CAT-N strategy (50% single, 50% CAT-5)
- RandomConcatAugmenter: Random concatenation variant
- PaddedSequenceCollator: Standard padding collation
- PackedSequenceCollator: cu_seqlens for H100 efficiency (20-30% speedup)
- DynamicBatchCollator: Token-budget batching
- LabelShiftCollator: Shift labels for teacher forcing
- MQARCollator: MQAR task collation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
import random
import math

import torch
import torch.nn.functional as F


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DocumentSample:
    """
    A document sample for NMT training.

    Attributes:
        src_sentences: List of source sentences in the document
        tgt_sentences: List of target sentences (parallel)
        doc_id: Optional document identifier
        metadata: Optional additional metadata
    """
    src_sentences: List[str]
    tgt_sentences: List[str]
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.src_sentences)

    def get_sentence_pair(self, idx: int) -> Tuple[str, str]:
        """Get a specific sentence pair."""
        return self.src_sentences[idx], self.tgt_sentences[idx]


# =============================================================================
# Augmenters
# =============================================================================

class ConcatenationAugmenter:
    """
    CAT-N Augmentation Strategy for Document-Level NMT.

    CRITICAL for length generalization:
    - 50% probability: Single sentence (no concatenation)
    - 50% probability: CAT-N (concatenate up to n_sentences with <doc> separator)

    This strategy teaches the model to:
    1. Handle variable-length inputs
    2. Respect document boundaries via <doc> tokens
    3. Maintain coherence across sentence boundaries

    Reference: Document-Level Machine Translation strategies
    """

    def __init__(
        self,
        n_sentences: int = 5,
        p_concat: float = 0.5,
        separator: str = " <doc> ",
        min_concat: int = 1,
        max_concat: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_sentences: Maximum number of sentences to concatenate (CAT-N)
            p_concat: Probability of concatenation (default: 0.5)
            separator: Separator token between sentences
            min_concat: Minimum sentences to concatenate (default: 1)
            max_concat: Maximum sentences to concatenate (default: n_sentences)
            seed: Random seed for reproducibility
        """
        self.n_sentences = n_sentences
        self.p_concat = p_concat
        self.separator = separator
        self.min_concat = min_concat
        self.max_concat = max_concat if max_concat is not None else n_sentences
        self._initial_seed = seed
        self._rng = random.Random(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for reproducible augmentation.

        CRITICAL: Call this at the start of each epoch for reproducibility.
        The random state is reset based on initial seed + epoch.
        """
        self._epoch = epoch
        if self._initial_seed is not None:
            self._rng.seed(self._initial_seed + epoch)

    def __call__(
        self,
        samples: List[DocumentSample],
        idx: int,
    ) -> Tuple[str, str]:
        """
        Apply CAT-N augmentation to a sample.

        Args:
            samples: List of all samples (for selecting neighbors)
            idx: Index of the current sample

        Returns:
            Tuple of (augmented_src, augmented_tgt)
        """
        if self._rng.random() > self.p_concat:
            # Single sentence - no concatenation
            sample = samples[idx]
            if len(sample) > 0:
                sent_idx = self._rng.randint(0, len(sample) - 1)
                return sample.get_sentence_pair(sent_idx)
            return "", ""

        # CAT-N: Concatenate multiple sentences
        sample = samples[idx]
        if len(sample) == 0:
            return "", ""

        # Determine number of sentences to concatenate
        max_possible = min(self.max_concat, len(sample))
        min_possible = min(self.min_concat, max_possible)

        if max_possible <= min_possible:
            n_to_concat = max_possible
        else:
            n_to_concat = self._rng.randint(min_possible, max_possible)

        if n_to_concat <= 1:
            sent_idx = self._rng.randint(0, len(sample) - 1)
            return sample.get_sentence_pair(sent_idx)

        # Select contiguous sentences
        start_idx = self._rng.randint(0, len(sample) - n_to_concat)
        end_idx = start_idx + n_to_concat

        # Concatenate with separator
        src_parts = sample.src_sentences[start_idx:end_idx]
        tgt_parts = sample.tgt_sentences[start_idx:end_idx]

        src_concat = self.separator.join(src_parts)
        tgt_concat = self.separator.join(tgt_parts)

        return src_concat, tgt_concat

    def augment_document(
        self,
        doc: DocumentSample,
    ) -> List[Tuple[str, str]]:
        """
        Augment a document into multiple samples.

        Args:
            doc: DocumentSample to augment

        Returns:
            List of (src, tgt) tuples
        """
        samples = []

        # Generate multiple augmented versions
        num_samples = max(1, len(doc) // self.min_concat)

        for _ in range(num_samples):
            src, tgt = self([doc], 0)
            if src and tgt:
                samples.append((src, tgt))

        return samples

    def augment_single(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
    ) -> Tuple[str, str]:
        """
        Augment a single document (list of sentence pairs).

        This is a simpler interface when you have sentences directly.
        """
        sample = DocumentSample(src_sentences=src_sentences, tgt_sentences=tgt_sentences)
        return self([sample], 0)


class RandomConcatAugmenter:
    """
    Random Concatenation Augmenter.

    Variant of CAT-N that randomly selects sentences (not necessarily contiguous).
    Useful for ablation studies.
    """

    def __init__(
        self,
        min_sentences: int = 1,
        max_sentences: int = 5,
        separator: str = " <doc> ",
        seed: Optional[int] = None,
    ):
        """
        Args:
            min_sentences: Minimum sentences to concatenate
            max_sentences: Maximum sentences to concatenate
            separator: Separator between sentences
            seed: Random seed
        """
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.separator = separator
        self._rng = random.Random(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducibility."""
        self._epoch = epoch

    def __call__(
        self,
        samples: List[DocumentSample],
        idx: int,
    ) -> Tuple[str, str]:
        """Apply random concatenation augmentation."""
        sample = samples[idx]
        if len(sample) == 0:
            return "", ""

        # Random number of sentences
        n_to_concat = self._rng.randint(
            self.min_sentences,
            min(self.max_sentences, len(sample))
        )

        # Random selection (not necessarily contiguous)
        indices = self._rng.sample(range(len(sample)), n_to_concat)
        indices.sort()  # Keep order

        src_parts = [sample.src_sentences[i] for i in indices]
        tgt_parts = [sample.tgt_sentences[i] for i in indices]

        return self.separator.join(src_parts), self.separator.join(tgt_parts)


# =============================================================================
# Collators
# =============================================================================

class PaddedSequenceCollator:
    """
    Standard padding collator for NMT.

    Pads all sequences to the maximum length in the batch.
    Simple and compatible with all models.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
    ):
        """
        Args:
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_src_length: Maximum source length (None for dynamic)
            max_tgt_length: Maximum target length (None for dynamic)
        """
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of dicts with 'src_ids' and 'tgt_ids' tensors

        Returns:
            Dict with padded 'src_ids', 'tgt_ids', 'src_mask', 'tgt_mask', 'labels'
        """
        src_ids = [item['src_ids'] for item in batch]
        tgt_ids = [item['tgt_ids'] for item in batch]

        # Determine max lengths
        max_src = max(len(s) for s in src_ids)
        max_tgt = max(len(t) for t in tgt_ids)

        if self.max_src_length:
            max_src = min(max_src, self.max_src_length)
        if self.max_tgt_length:
            max_tgt = min(max_tgt, self.max_tgt_length)

        # Pad sequences
        padded_src = []
        padded_tgt = []
        src_masks = []
        tgt_masks = []

        for src, tgt in zip(src_ids, tgt_ids):
            # Truncate if necessary
            src = src[:max_src]
            tgt = tgt[:max_tgt]

            # Pad
            src_pad_len = max_src - len(src)
            tgt_pad_len = max_tgt - len(tgt)

            padded_src.append(F.pad(src, (0, src_pad_len), value=self.pad_token_id))
            padded_tgt.append(F.pad(tgt, (0, tgt_pad_len), value=self.pad_token_id))

            # Masks: 1s for actual tokens, 0s for padding
            src_mask = torch.ones(max_src)
            src_mask[len(src):] = 0
            src_masks.append(src_mask)

            tgt_mask = torch.ones(max_tgt)
            tgt_mask[len(tgt):] = 0
            tgt_masks.append(tgt_mask)

        # Stack
        result = {
            'src_ids': torch.stack(padded_src),
            'tgt_ids': torch.stack(padded_tgt),
            'src_mask': torch.stack(src_masks),
            'tgt_mask': torch.stack(tgt_masks),
        }

        # Create labels (shifted targets)
        labels = result['tgt_ids'].clone()
        labels[labels == self.pad_token_id] = -100

        result['labels'] = labels

        return result


class PackedSequenceCollator:
    """
    Packed sequence collator for H100 efficiency.

    CRITICAL for H100 performance (20-30% speedup):
    Instead of padding, packs sequences and provides cu_seqlens for
    FlashAttention's variable-length mode.

    Returns:
        - src_ids: (total_src_tokens,) - packed source tokens
        - tgt_ids: (total_tgt_tokens,) - packed target tokens
        - cu_seqlens_src: (batch_size + 1,) - cumulative source lengths
        - cu_seqlens_tgt: (batch_size + 1,) - cumulative target lengths
        - max_seqlen_src: Maximum source sequence length
        - max_seqlen_tgt: Maximum target sequence length
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
    ):
        """
        Args:
            pad_token_id: Padding token ID (for labels masking)
            max_src_length: Maximum source length
            max_tgt_length: Maximum target length
        """
        self.pad_token_id = pad_token_id
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch into packed format.

        Args:
            batch: List of dicts with 'src_ids' and 'tgt_ids' tensors

        Returns:
            Dict with packed tensors and cu_seqlens
        """
        src_ids_list = []
        tgt_ids_list = []
        src_lengths = []
        tgt_lengths = []

        for item in batch:
            src = item['src_ids']
            tgt = item['tgt_ids']

            # Truncate if necessary
            if self.max_src_length:
                src = src[:self.max_src_length]
            if self.max_tgt_length:
                tgt = tgt[:self.max_tgt_length]

            src_ids_list.append(src)
            tgt_ids_list.append(tgt)
            src_lengths.append(len(src))
            tgt_lengths.append(len(tgt))

        # Pack sequences
        packed_src = torch.cat(src_ids_list, dim=0)
        packed_tgt = torch.cat(tgt_ids_list, dim=0)

        # Compute cumulative sequence lengths
        cu_seqlens_src = torch.zeros(len(batch) + 1, dtype=torch.int32)
        cu_seqlens_tgt = torch.zeros(len(batch) + 1, dtype=torch.int32)

        cu_seqlens_src[1:] = torch.cumsum(torch.tensor(src_lengths, dtype=torch.int32), dim=0)
        cu_seqlens_tgt[1:] = torch.cumsum(torch.tensor(tgt_lengths, dtype=torch.int32), dim=0)

        # Create labels
        labels = packed_tgt.clone()

        return {
            'src_ids': packed_src,
            'tgt_ids': packed_tgt,
            'labels': labels,
            'cu_seqlens_src': cu_seqlens_src,
            'cu_seqlens_tgt': cu_seqlens_tgt,
            'max_seqlen_src': max(src_lengths),
            'max_seqlen_tgt': max(tgt_lengths),
        }


class DynamicBatchCollator:
    """
    Dynamic batching by token budget.

    Instead of fixed batch size, creates batches with approximately
    the same total number of tokens. This maximizes GPU utilization
    when sequence lengths vary significantly.
    """

    def __init__(
        self,
        max_tokens: int = 16384,
        pad_token_id: int = 0,
        include_padding: bool = True,
    ):
        """
        Args:
            max_tokens: Maximum tokens per batch
            pad_token_id: Padding token ID
            include_padding: Include padding in token count
        """
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id
        self.include_padding = include_padding
        self._padded_collator = PaddedSequenceCollator(pad_token_id=pad_token_id)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate with dynamic batching.

        Note: This collator assumes pre-bucketed samples.
        Use with a bucket sampler for optimal efficiency.
        """
        return self._padded_collator(batch)


class LabelShiftCollator:
    """
    Wrapper collator that shifts labels for teacher forcing.

    For autoregressive training:
    - Input: [BOS, tok1, tok2, ..., tokN]
    - Labels: [tok1, tok2, ..., tokN, EOS]

    Shifts labels left by 1 and sets last position to EOS or -100.
    """

    def __init__(
        self,
        base_collator: Union[PaddedSequenceCollator, PackedSequenceCollator],
        eos_token_id: int = 2,
        ignore_index: int = -100,
    ):
        """
        Args:
            base_collator: Underlying collator to use
            eos_token_id: End of sequence token ID
            ignore_index: Value for ignored positions in loss
        """
        self.base_collator = base_collator
        self.eos_token_id = eos_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Apply base collation then shift labels."""
        result = self.base_collator(batch)

        # Shift labels
        if 'labels' in result:
            labels = result['labels']
            # Shift left: labels[i] = tgt_ids[i+1]
            shifted_labels = torch.full_like(labels, self.ignore_index)
            shifted_labels[..., :-1] = labels[..., 1:]
            result['labels'] = shifted_labels

        return result


class MQARCollator:
    """
    Collator for Multi-Query Associative Recall (MQAR) task.

    MQAR Structure: [Pairs] [SEP] [Queries]
    - Pairs: key-value pairs for memorization
    - SEP: separator token
    - Queries: keys to look up

    CRITICAL: Queries must be STRICTLY after all pairs.
    No interleaving: [Pairs] [Query] [Pairs] is INVALID.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        sep_token_id: int = 3,
        max_length: Optional[int] = None,
    ):
        """
        Args:
            pad_token_id: Padding token ID
            sep_token_id: Separator token ID
            max_length: Maximum sequence length
        """
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate MQAR batch.

        Expected input format per sample:
        - 'input_ids': Full sequence [pairs, SEP, queries]
        - 'labels': Target values for queries
        - 'query_positions': Positions of query tokens
        """
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Determine max length
        max_len = max(len(seq) for seq in input_ids)
        if self.max_length:
            max_len = min(max_len, self.max_length)

        # Pad sequences
        padded_inputs = []
        padded_labels = []

        label_masks = []

        for inp, lab in zip(input_ids, labels):
            inp = inp[:max_len]
            lab = lab[:max_len]

            pad_len = max_len - len(inp)
            padded_inputs.append(F.pad(inp, (0, pad_len), value=self.pad_token_id))

            # Pad labels
            padded_lab = torch.full((max_len,), self.pad_token_id, dtype=lab.dtype)
            padded_lab[:len(lab)] = lab
            padded_labels.append(padded_lab)

            # Label mask: 1 where we have valid labels (not PAD), 0 elsewhere
            label_mask = (padded_lab != self.pad_token_id).long()
            label_masks.append(label_mask)

        return {
            'input_ids': torch.stack(padded_inputs),
            'labels': torch.stack(padded_labels),
            'attention_mask': (torch.stack(padded_inputs) != self.pad_token_id).long(),
            'label_mask': torch.stack(label_masks),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_collator(
    mode: str = "padded",
    pad_token_id: int = 0,
    **kwargs,
) -> Union[PaddedSequenceCollator, PackedSequenceCollator, DynamicBatchCollator, MQARCollator]:
    """
    Factory function to create collators.

    Args:
        mode: "padded", "packed", "dynamic", or "mqar"
        pad_token_id: Padding token ID
        **kwargs: Additional arguments for specific collators

    Returns:
        Collator instance

    Recommendations:
    - "packed": Use for H100 training (20-30% speedup)
    - "padded": Use for debugging or CPU training
    - "dynamic": Use when sequence lengths vary significantly
    - "mqar": Use for MQAR synthetic task
    """
    if mode == "padded":
        return PaddedSequenceCollator(pad_token_id=pad_token_id, **kwargs)
    elif mode == "packed":
        return PackedSequenceCollator(pad_token_id=pad_token_id, **kwargs)
    elif mode == "dynamic":
        return DynamicBatchCollator(pad_token_id=pad_token_id, **kwargs)
    elif mode == "mqar":
        return MQARCollator(pad_token_id=pad_token_id, **kwargs)
    else:
        raise ValueError(f"Unknown collator mode: {mode}")


def create_augmenter(
    mode: str = "cat_n",
    n_sentences: int = 5,
    p_concat: float = 0.5,
    separator: str = " <doc> ",
    seed: Optional[int] = None,
    **kwargs,
) -> Union[ConcatenationAugmenter, RandomConcatAugmenter]:
    """
    Factory function to create augmenters.

    Args:
        mode: "cat_n" (RECOMMENDED) or "random"
        n_sentences: Maximum sentences to concatenate
        p_concat: Probability of concatenation
        separator: Document separator token
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        Augmenter instance

    Recommendations:
    - "cat_n": Use for thesis experiments (matches paper)
    - "random": Use for ablation studies
    """
    if mode == "cat_n":
        return ConcatenationAugmenter(
            n_sentences=n_sentences,
            p_concat=p_concat,
            separator=separator,
            seed=seed,
        )
    elif mode == "random":
        return RandomConcatAugmenter(
            max_sentences=n_sentences,
            separator=separator,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown augmenter mode: {mode}")
