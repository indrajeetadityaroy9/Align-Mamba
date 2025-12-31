"""
Custom Vocab class to replace deprecated torchtext.vocab.

This module provides a vocabulary implementation compatible with the NMT pipeline,
supporting token-to-index mapping, special tokens, and serialization.
"""

from collections import Counter, OrderedDict
from typing import List, Dict, Iterator
import torch


class Vocab:
    """
    Vocabulary class for mapping tokens to indices and vice versa.

    Replaces torchtext.vocab.Vocab which is deprecated in PyTorch 2.4+.

    Attributes:
        token_to_idx: Mapping from tokens to integer indices
        idx_to_token: Mapping from indices to tokens
        default_index: Index returned for unknown tokens
    """

    def __init__(self, token_to_idx: Dict[str, int], default_index: int = 1):
        """
        Initialize vocabulary with token-to-index mapping.

        Args:
            token_to_idx: Dictionary mapping tokens to indices
            default_index: Index to return for OOV tokens (default: 1 for <unk>)
        """
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        self.default_index = default_index

    def __getitem__(self, token: str) -> int:
        """Get index for a token, returns default_index if not found."""
        return self.token_to_idx.get(token, self.default_index)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_idx)

    def __call__(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to list of indices."""
        return [self[token] for token in tokens]

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token_to_idx

    def lookup_token(self, idx: int) -> str:
        """Convert index back to token."""
        return self.idx_to_token.get(idx, '<unk>')

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        """Convert list of indices to list of tokens."""
        return [self.lookup_token(idx) for idx in indices]

    def set_default_index(self, index: int) -> None:
        """Set the default index for unknown tokens."""
        self.default_index = index

    def get_stoi(self) -> Dict[str, int]:
        """Get string-to-index mapping (for compatibility)."""
        return self.token_to_idx

    def get_itos(self) -> List[str]:
        """Get index-to-string list (for compatibility)."""
        return [self.idx_to_token[i] for i in range(len(self))]


def build_vocab_from_iterator(
    iterator: Iterator[List[str]],
    specials: List[str] = None,
    special_first: bool = True,
    min_freq: int = 1
) -> Vocab:
    """
    Build vocabulary from an iterator of tokenized texts.

    This is a drop-in replacement for torchtext.vocab.build_vocab_from_iterator.

    Args:
        iterator: Iterator yielding lists of tokens
        specials: List of special tokens (e.g., ['<pad>', '<unk>', '<bos>', '<eos>'])
        special_first: If True, special tokens get lowest indices
        min_freq: Minimum frequency for a token to be included

    Returns:
        Vocab object with token-to-index mappings

    Example:
        >>> texts = [['hello', 'world'], ['hello', 'there']]
        >>> vocab = build_vocab_from_iterator(
        ...     iter(texts),
        ...     specials=['<pad>', '<unk>', '<bos>', '<eos>']
        ... )
        >>> vocab['hello']
        4
        >>> vocab['<unk>']
        1
    """
    if specials is None:
        specials = []

    # Count token frequencies
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    # Build ordered token-to-index mapping
    token_to_idx = OrderedDict()

    if special_first:
        # Add special tokens first with lowest indices
        for i, token in enumerate(specials):
            token_to_idx[token] = i

    # Add regular tokens sorted by frequency (descending)
    for token, freq in counter.most_common():
        if token not in token_to_idx and freq >= min_freq:
            token_to_idx[token] = len(token_to_idx)

    if not special_first:
        # Add special tokens at the end
        for token in specials:
            if token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)

    # Create vocab with <unk> as default (index 1 if specials follow standard order)
    default_index = token_to_idx.get('<unk>', 0)
    vocab = Vocab(token_to_idx, default_index=default_index)

    return vocab
