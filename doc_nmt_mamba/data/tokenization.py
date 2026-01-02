"""
Tokenization for Document-Level NMT.

Supports two modes:
1. Custom 32K BPE tokenizer (RECOMMENDED for thesis - proper parameter allocation)
2. mBART tokenizer (250K vocab - NOT recommended, 95% embedding table)

CRITICAL: Use CustomBPETokenizer for scientific validity.
"""

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import MBart50TokenizerFast


class CustomBPETokenizer:
    """
    Custom 32K BPE tokenizer for Document-Level NMT.

    RECOMMENDED: This tokenizer uses a proper 32K vocabulary, ensuring
    the model's parameters are allocated to compute layers rather than
    a massive embedding table.

    Features:
    - 32K vocabulary (comparable to "Attention Is All You Need")
    - Byte-level BPE (handles any unicode)
    - Fast Rust implementation via HuggingFace tokenizers
    """

    def __init__(
        self,
        tokenizer_path: str = "data/tokenizer/tokenizer.json",
        max_length: int = 8192,
    ):
        """
        Args:
            tokenizer_path: Path to tokenizer.json file
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.tokenizer_path = Path(tokenizer_path)

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                f"Run: python scripts/build_tokenizer.py"
            )

        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        # Get special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")

        # Verify special tokens exist
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
        """
        HuggingFace-compatible interface for encoding text.

        Args:
            text: Text or list of texts to encode
            padding: Whether to pad sequences (default: True)
            truncation: Whether to truncate sequences (default: True)
            max_length: Maximum sequence length
            return_tensors: "pt" for PyTorch tensors, None for lists

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
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
        """Encode source text."""
        max_length = max_length or self.max_length

        if isinstance(text, str):
            text = [text]

        # Enable truncation for this encoding
        self.tokenizer.enable_truncation(max_length=max_length)

        encoded = self.tokenizer.encode_batch(text)

        input_ids = [e.ids for e in encoded]
        attention_mask = [[1] * len(ids) for ids in input_ids]

        if return_tensors:
            # Pad to max length in batch
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
        """Encode target text (same as source for shared vocab)."""
        return self.encode_source(text, max_length, return_tensors)

    def encode_pair(
        self,
        src_text: str,
        tgt_text: str,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode source-target pair."""
        src_encoded = self.encode_source(src_text, max_src_length)
        tgt_encoded = self.encode_target(tgt_text, max_tgt_length)

        return src_encoded["input_ids"].squeeze(0), tgt_encoded["input_ids"].squeeze(0)

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Filter out padding if skipping special tokens
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode batch of token IDs."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return [self.decode(ids, skip_special_tokens) for ids in token_ids]


class NMTTokenizer:
    """
    Wrapper around mBART tokenizer for NMT.

    Features:
    - Source/target language codes
    - Efficient batch encoding
    - Document boundary handling
    """

    def __init__(
        self,
        src_lang: str = "de_DE",
        tgt_lang: str = "en_XX",
        model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
        max_length: int = 8192,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            src_lang: Source language code (mBART format)
            tgt_lang: Target language code (mBART format)
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
            cache_dir: Cache directory for tokenizer files
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            model_name,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            cache_dir=cache_dir,
        )

        # Store special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        # Language tokens
        self.src_lang_id = self.tokenizer.lang_code_to_id[src_lang]
        self.tgt_lang_id = self.tokenizer.lang_code_to_id[tgt_lang]

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode_source(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        return_tensors: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode source text.

        Args:
            text: Source text or list of texts
            max_length: Maximum length (uses default if None)
            return_tensors: Return PyTorch tensors

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        max_length = max_length or self.max_length

        # Set source language
        self.tokenizer.src_lang = self.src_lang

        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=True if isinstance(text, list) else False,
            return_tensors="pt" if return_tensors else None,
        )

        return encoded

    def encode_target(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        return_tensors: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode target text.

        Args:
            text: Target text or list of texts
            max_length: Maximum length (uses default if None)
            return_tensors: Return PyTorch tensors

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        max_length = max_length or self.max_length

        # Set target language for tokenization
        self.tokenizer.tgt_lang = self.tgt_lang

        # Use target language as prefix
        with self.tokenizer.as_target_tokenizer():
            encoded = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=True if isinstance(text, list) else False,
                return_tensors="pt" if return_tensors else None,
            )

        return encoded

    def encode_pair(
        self,
        src_text: str,
        tgt_text: str,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode source-target pair.

        Args:
            src_text: Source text
            tgt_text: Target text
            max_src_length: Max source length
            max_tgt_length: Max target length

        Returns:
            Tuple of (src_ids, tgt_ids)
        """
        src_encoded = self.encode_source(src_text, max_src_length)
        tgt_encoded = self.encode_target(tgt_text, max_tgt_length)

        return src_encoded["input_ids"].squeeze(0), tgt_encoded["input_ids"].squeeze(0)

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode batch of token IDs.

        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Skip special tokens

        Returns:
            List of decoded texts
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)


def create_tokenizer(
    tokenizer_type: str = "custom",
    tokenizer_path: Optional[str] = None,
    src_lang: str = "de_DE",
    tgt_lang: str = "en_XX",
    **kwargs,
) -> Union[CustomBPETokenizer, "NMTTokenizer"]:
    """
    Factory function to create tokenizer.

    Args:
        tokenizer_type: "custom" (RECOMMENDED, 32K vocab) or "mbart" (250K vocab)
        tokenizer_path: Path to custom tokenizer.json (for custom type)
        src_lang: Source language (for mbart type)
        tgt_lang: Target language (for mbart type)
        **kwargs: Additional tokenizer arguments

    Returns:
        Tokenizer instance

    IMPORTANT: Use tokenizer_type="custom" for thesis work.
    The mBART tokenizer has 250K vocab which makes the model 95% embedding table.
    """
    if tokenizer_type == "custom":
        path = tokenizer_path or "data/tokenizer/tokenizer.json"
        return CustomBPETokenizer(tokenizer_path=path, **kwargs)
    elif tokenizer_type == "mbart":
        import warnings
        warnings.warn(
            "Using mBART tokenizer (250K vocab). "
            "This makes the model 95% embedding table. "
            "Use tokenizer_type='custom' for thesis work."
        )
        return NMTTokenizer(src_lang=src_lang, tgt_lang=tgt_lang, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
