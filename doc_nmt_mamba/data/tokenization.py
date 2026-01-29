"""Tokenization for document-level NMT using 32K BPE."""

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import torch
from tokenizers import Tokenizer


class CustomBPETokenizer:
    """32K BPE tokenizer - keeps parameters in compute layers rather than massive embedding table."""

    def __init__(
        self,
        tokenizer_path: str = "data/tokenizer/tokenizer.json",
        max_length: int = 8192,
    ):
        self.max_length = max_length
        self.tokenizer_path = Path(tokenizer_path)

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                f"Run: python scripts/build_tokenizer.py"
            )

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")

        assert self.pad_token_id is not None, "Missing <pad> token"
        assert self.bos_token_id is not None, "Missing <s> token"
        assert self.eos_token_id is not None, "Missing </s> token"

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        return self.encode_source(
            text,
            max_length=max_length,
            return_tensors=(return_tensors == "pt"),
        )

    def encode_source(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        return_tensors: bool = True,
    ) -> Dict[str, torch.Tensor]:
        max_length = max_length or self.max_length

        if isinstance(text, str):
            text = [text]

        self.tokenizer.enable_truncation(max_length=max_length)

        encoded = self.tokenizer.encode_batch(text)

        input_ids = [e.ids for e in encoded]
        attention_mask = [[1] * len(ids) for ids in input_ids]

        if return_tensors:
            max_len = max(len(ids) for ids in input_ids)
            input_ids = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
            attention_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]

            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_target(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        return_tensors: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self.encode_source(text, max_length, return_tensors)

    def encode_pair(
        self,
        src_text: str,
        tgt_text: str,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_encoded = self.encode_source(src_text, max_src_length)
        tgt_encoded = self.encode_target(tgt_text, max_tgt_length)

        return src_encoded["input_ids"].squeeze(0), tgt_encoded["input_ids"].squeeze(0)

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return [self.decode(ids, skip_special_tokens) for ids in token_ids]


def create_tokenizer(
    tokenizer_path: Optional[str] = None,
    **kwargs,
) -> CustomBPETokenizer:
    path = tokenizer_path or "data/tokenizer/tokenizer.json"
    return CustomBPETokenizer(tokenizer_path=path, **kwargs)
