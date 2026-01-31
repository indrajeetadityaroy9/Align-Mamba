"""
MQAR (Multi-Query Associative Recall) synthetic task for state capacity testing.
With d_state=64, creates a "state capacity cliff" when num_pairs > d_state.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import random

import torch
from torch.utils.data import Dataset

import torch.nn.functional as F

from ..config import (
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID, QUERY_TOKEN_ID,
    KEY_TOKEN_START, KEY_TOKEN_END, VALUE_TOKEN_START, VALUE_TOKEN_END,
    MQAR_VOCAB_SIZE, MQAR_SEQ_LENGTH,
)
from typing import List


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
            print(
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
            }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._samples[idx]


class MQARCollator:
    """MQAR task collator with two modes:

    - decoder_only: For pure Mamba (TC0) - concatenated input/output
    - seq2seq: For Hybrid (NC1) - encoder gets pairs, decoder gets queries
    """

    def __init__(
        self,
        pad_token_id: int = PAD_TOKEN_ID,
        max_length: Optional[int] = None,
        mode: str = "decoder_only",
    ):
        if mode not in ("decoder_only", "seq2seq"):
            raise ValueError(f"mode must be 'decoder_only' or 'seq2seq', got '{mode}'")

        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.mode = mode

    def _pad_sequences(
        self,
        sequences: List[torch.Tensor],
        pad_value: int,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pad sequences to max length."""
        if max_len is None:
            max_len = max(len(s) for s in sequences)
        if self.max_length:
            max_len = min(max_len, self.max_length)

        padded = []
        for seq in sequences:
            seq = seq[:max_len]
            pad_len = max_len - len(seq)
            padded.append(F.pad(seq, (0, pad_len), value=pad_value))

        return torch.stack(padded)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if self.mode == "seq2seq":
            return self._collate_seq2seq(batch)
        return self._collate_decoder_only(batch)

    def _collate_seq2seq(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Labels aligned for teacher forcing."""
        src_ids = self._pad_sequences([item['src_ids'] for item in batch], self.pad_token_id)
        tgt_ids = self._pad_sequences([item['tgt_ids'] for item in batch], self.pad_token_id)
        labels = self._pad_sequences([item['labels'] for item in batch], -100, max_len=tgt_ids.size(1) - 1)

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'labels': labels,
            'src_mask': (src_ids != self.pad_token_id).long(),
            'tgt_mask': (tgt_ids != self.pad_token_id).long(),
        }

    def _collate_decoder_only(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = self._pad_sequences([item['input_ids'] for item in batch], self.pad_token_id)
        labels = self._pad_sequences([item['labels'] for item in batch], -100, max_len=input_ids.size(1))

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.pad_token_id).long(),
        }
