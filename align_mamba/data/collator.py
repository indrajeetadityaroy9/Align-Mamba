"""MQAR collation for State Capacity experiments."""

from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

from ..constants import PAD_TOKEN_ID, SEP_TOKEN_ID


class MQARCollator:
    """
    MQAR task collator with two modes:
    - decoder_only: For pure Mamba (TC0) - concatenated input/output
    - seq2seq: For Hybrid (NC1) - encoder gets pairs, decoder gets queries
    """

    def __init__(
        self,
        pad_token_id: int = PAD_TOKEN_ID,
        sep_token_id: int = SEP_TOKEN_ID,
        max_length: Optional[int] = None,
        mode: str = "decoder_only",
    ):
        if mode not in ("decoder_only", "seq2seq"):
            raise ValueError(f"mode must be 'decoder_only' or 'seq2seq', got '{mode}'")

        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length
        self.mode = mode

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if self.mode == "seq2seq":
            return self._collate_seq2seq(batch)
        else:
            return self._collate_decoder_only(batch)

    def _collate_seq2seq(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Labels aligned for teacher forcing."""
        src_ids_list = [item['src_ids'] for item in batch]
        tgt_ids_list = [item['tgt_ids'] for item in batch]
        labels_list = [item['labels'] for item in batch]

        max_src_len = max(len(seq) for seq in src_ids_list)
        max_tgt_len = max(len(seq) for seq in tgt_ids_list)

        if self.max_length:
            max_src_len = min(max_src_len, self.max_length)
            max_tgt_len = min(max_tgt_len, self.max_length)

        max_labels_len = max_tgt_len - 1

        padded_src = []
        padded_tgt = []
        padded_labels = []

        for src, tgt, lab in zip(src_ids_list, tgt_ids_list, labels_list):
            src = src[:max_src_len]
            tgt = tgt[:max_tgt_len]
            lab = lab[:max_labels_len]

            src_pad_len = max_src_len - len(src)
            padded_src.append(F.pad(src, (0, src_pad_len), value=self.pad_token_id))

            tgt_pad_len = max_tgt_len - len(tgt)
            padded_tgt.append(F.pad(tgt, (0, tgt_pad_len), value=self.pad_token_id))

            padded_lab = torch.full((max_labels_len,), -100, dtype=lab.dtype)
            padded_lab[:len(lab)] = lab
            padded_labels.append(padded_lab)

        src_tensor = torch.stack(padded_src)
        tgt_tensor = torch.stack(padded_tgt)
        labels_tensor = torch.stack(padded_labels)

        return {
            'src_ids': src_tensor,
            'tgt_ids': tgt_tensor,
            'labels': labels_tensor,
            'src_mask': (src_tensor != self.pad_token_id).long(),
            'tgt_mask': (tgt_tensor != self.pad_token_id).long(),
        }

    def _collate_decoder_only(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        max_len = max(len(seq) for seq in input_ids)
        if self.max_length:
            max_len = min(max_len, self.max_length)

        padded_inputs = []
        padded_labels = []

        for inp, lab in zip(input_ids, labels):
            inp = inp[:max_len]
            lab = lab[:max_len]

            pad_len = max_len - len(inp)
            padded_inputs.append(F.pad(inp, (0, pad_len), value=self.pad_token_id))

            padded_lab = torch.full((max_len,), -100, dtype=lab.dtype)
            padded_lab[:len(lab)] = lab
            padded_labels.append(padded_lab)

        input_tensor = torch.stack(padded_inputs)

        return {
            'input_ids': input_tensor,
            'labels': torch.stack(padded_labels),
            'attention_mask': (input_tensor != self.pad_token_id).long(),
        }


def create_collator(mode: str = "decoder_only", **kwargs) -> MQARCollator:
    """Factory for MQAR collator."""
    return MQARCollator(mode=mode, **kwargs)
