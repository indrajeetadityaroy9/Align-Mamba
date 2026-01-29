"""
MQAR (Multi-Query Associative Recall) synthetic task for state capacity testing.
With d_state=64, creates a "state capacity cliff" when num_pairs > d_state.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import random
import logging

import torch
from torch.utils.data import Dataset

from ..constants import (
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID, QUERY_TOKEN_ID,
    KEY_TOKEN_START, KEY_TOKEN_END, VALUE_TOKEN_START, VALUE_TOKEN_END,
    MQAR_VOCAB_SIZE, MQAR_SEQ_LENGTH,
)

logger = logging.getLogger(__name__)


@dataclass
class MQARConfig:
    """
    MQAR task config - only experiment-varying parameters.
    Infrastructure values (vocab_size, seq_length, token IDs) use constants.
    """
    num_pairs: int = 64
    num_queries: int = 16


class MQARDataset(Dataset):
    """
    Synthetic MQAR dataset. Keys are unique, queries are a subset of keys.

    Modes:
    - decoder_only: Concatenated for pure Mamba (TC0)
    - seq2seq: Split for Hybrid cross-attention (NC1)
    """

    def __init__(
        self,
        config: MQARConfig,
        num_samples: int = 10000,
        split: str = "train",
        mode: str = "decoder_only",
    ):
        if mode not in ("decoder_only", "seq2seq"):
            raise ValueError(f"mode must be 'decoder_only' or 'seq2seq', got '{mode}'")

        self.config = config
        self.num_samples = num_samples
        self.split = split
        self.mode = mode
        self.seq_length = MQAR_SEQ_LENGTH

        # Validate seq_length can fit all tokens
        num_queries = min(config.num_queries, config.num_pairs)
        min_seq_len = 1 + (config.num_pairs * 3) + 1 + num_queries + 1

        if self.seq_length < min_seq_len:
            self.seq_length = int(min_seq_len * 1.1)
            logger.warning(
                f"seq_length={MQAR_SEQ_LENGTH} too short for num_pairs={config.num_pairs}. "
                f"Auto-extended to {self.seq_length} (min required: {min_seq_len})"
            )

        # Validate key range can provide unique keys
        key_range_size = KEY_TOKEN_END - KEY_TOKEN_START
        if key_range_size < config.num_pairs:
            raise ValueError(
                f"Key range [{KEY_TOKEN_START}, {KEY_TOKEN_END}) "
                f"has only {key_range_size} unique keys, but num_pairs={config.num_pairs} required."
            )

        self._seed = 42
        self._rng = random.Random(42)

        if split == "validation":
            self._rng.seed(self._seed + 1000)
        elif split == "test":
            self._rng.seed(self._seed + 2000)

        self._samples = [self._generate_sample(i) for i in range(num_samples)]

    def _generate_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        key_range = range(KEY_TOKEN_START, KEY_TOKEN_END)
        keys = self._rng.sample(list(key_range), self.config.num_pairs)

        value_range = range(VALUE_TOKEN_START, VALUE_TOKEN_END)
        values = [self._rng.choice(list(value_range)) for _ in range(self.config.num_pairs)]

        kv_map = dict(zip(keys, values))

        query_keys = self._rng.sample(keys, min(self.config.num_queries, len(keys)))

        if self.mode == "seq2seq":
            # Encoder gets context, decoder gets queries
            src_ids = [BOS_TOKEN_ID]
            for k, v in zip(keys, values):
                src_ids.append(k)
                src_ids.append(SEP_TOKEN_ID)
                src_ids.append(v)
            src_ids.append(EOS_TOKEN_ID)

            tgt_ids = [BOS_TOKEN_ID]
            for qk in query_keys:
                tgt_ids.append(qk)
            tgt_ids.append(EOS_TOKEN_ID)

            # Labels: -100 for BOS, then values aligned with queries
            labels = [-100]
            for qk in query_keys:
                labels.append(kv_map[qk])

            max_src_len = 1 + (self.config.num_pairs * 3) + 1
            if len(src_ids) < max_src_len:
                src_ids.extend([PAD_TOKEN_ID] * (max_src_len - len(src_ids)))

            max_tgt_len = 1 + self.config.num_queries + 1
            if len(tgt_ids) < max_tgt_len:
                tgt_ids.extend([PAD_TOKEN_ID] * (max_tgt_len - len(tgt_ids)))

            max_labels_len = max_tgt_len - 1
            if len(labels) < max_labels_len:
                labels.extend([-100] * (max_labels_len - len(labels)))

            return {
                'src_ids': torch.tensor(src_ids, dtype=torch.long),
                'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'query_keys': torch.tensor(query_keys, dtype=torch.long),
                'expected_values': torch.tensor([kv_map[k] for k in query_keys], dtype=torch.long),
                'num_pairs': torch.tensor(self.config.num_pairs, dtype=torch.long),
                'num_queries': torch.tensor(len(query_keys), dtype=torch.long),
            }

        else:
            # Decoder-only: full concatenated sequence
            input_ids = [BOS_TOKEN_ID]

            for k, v in zip(keys, values):
                input_ids.append(k)
                input_ids.append(SEP_TOKEN_ID)
                input_ids.append(v)

            input_ids.append(QUERY_TOKEN_ID)

            query_positions = []
            for qk in query_keys:
                query_positions.append(len(input_ids))
                input_ids.append(qk)

            input_ids.append(EOS_TOKEN_ID)

            # Validate query positions are within bounds
            assert len(query_positions) == len(query_keys), "Position/key count mismatch"
            assert all(0 <= p < len(input_ids) for p in query_positions), \
                f"Query position out of bounds: {query_positions}, seq_len={len(input_ids)}"

            # Labels: -100 everywhere except query positions get expected values
            labels = [-100] * len(input_ids)
            for pos, qk in zip(query_positions, query_keys):
                labels[pos] = kv_map[qk]

            if len(input_ids) < self.seq_length:
                padding_len = self.seq_length - len(input_ids)
                input_ids.extend([PAD_TOKEN_ID] * padding_len)
                labels.extend([-100] * padding_len)
            elif len(input_ids) > self.seq_length:
                input_ids = input_ids[:self.seq_length]
                labels = labels[:self.seq_length]

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'query_positions': torch.tensor(query_positions, dtype=torch.long),
                'query_keys': torch.tensor(query_keys, dtype=torch.long),
                'expected_values': torch.tensor([kv_map[k] for k in query_keys], dtype=torch.long),
                'num_pairs': torch.tensor(self.config.num_pairs, dtype=torch.long),
                'num_queries': torch.tensor(len(query_keys), dtype=torch.long),
            }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._samples[idx]


def compute_mqar_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute token and sample accuracy at masked positions."""
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)

    if label_mask is not None:
        valid_mask = label_mask.bool()
    else:
        valid_mask = labels != -100

    if valid_mask.sum() == 0:
        return {'token_accuracy': 0.0, 'sample_accuracy': 0.0}

    correct = (predictions == labels) & valid_mask
    token_accuracy = correct.sum().float() / valid_mask.sum().float()

    correct_per_sample = correct.sum(dim=-1)
    total_per_sample = valid_mask.sum(dim=-1)

    has_labels = total_per_sample > 0
    perfect_samples = torch.zeros_like(total_per_sample, dtype=torch.float)
    if has_labels.any():
        perfect_samples[has_labels] = (
            correct_per_sample[has_labels] == total_per_sample[has_labels]
        ).float()
    sample_accuracy = perfect_samples.mean() if has_labels.any() else 0.0

    return {
        'token_accuracy': token_accuracy.item(),
        'sample_accuracy': sample_accuracy.item() if isinstance(sample_accuracy, torch.Tensor) else sample_accuracy,
    }
