"""Multi30k dataset and batching utilities."""

from typing import Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from data.tokenizer import tokenize, tokenize_german, tokenize_english
from data.vocab import Vocab, build_vocab_from_iterator

if TYPE_CHECKING:
    from data.subword_tokenizer import SubwordTokenizer


def load_multi30k():
    """Load Multi30k German-English dataset from HuggingFace."""
    print("Loading Multi30k De-En dataset...")
    return load_dataset("bentrevett/multi30k")


class Multi30kDataset(Dataset):
    """Dataset for Multi30k De-En translation pairs."""

    def __init__(self, hf_dataset, max_seq_length: int = 100):
        self.pairs = []
        for example in hf_dataset:
            src = example['de']
            tgt = example['en']

            src_tokens = tokenize_german(src)[:max_seq_length]
            tgt_tokens = tokenize_english(tgt)[:max_seq_length]

            if len(src_tokens) > 0 and len(tgt_tokens) > 0:
                self.pairs.append((' '.join(src_tokens), ' '.join(tgt_tokens)))

        self.src_sequences = [p[0] for p in self.pairs]
        self.tgt_sequences = [p[1] for p in self.pairs]

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)


def build_vocabulary(texts: list[str], lang: str) -> Vocab:
    """Build vocabulary from texts with special tokens."""
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']

    def token_iterator():
        for text in texts:
            yield tokenize(text, lang)

    vocabulary = build_vocab_from_iterator(
        token_iterator(),
        specials=special_tokens,
        special_first=True
    )
    vocabulary.set_default_index(vocabulary['<unk>'])
    return vocabulary


def collate_batch(batch, src_vocab: Vocab, tgt_vocab: Vocab,
                  src_lang: str, tgt_lang: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate batch of translation pairs into padded tensors with lengths."""
    src_batch, tgt_batch, src_lengths = [], [], []

    for src_text, tgt_text in batch:
        src_tokens = ['<bos>'] + tokenize(src_text, src_lang) + ['<eos>']
        tgt_tokens = ['<bos>'] + tokenize(tgt_text, tgt_lang) + ['<eos>']

        src_indices = torch.tensor(src_vocab(src_tokens), dtype=torch.long)
        tgt_indices = torch.tensor(tgt_vocab(tgt_tokens), dtype=torch.long)

        src_batch.append(src_indices)
        tgt_batch.append(tgt_indices)
        src_lengths.append(len(src_tokens))

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab['<pad>'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab['<pad>'])
    src_lengths = torch.tensor(src_lengths, dtype=torch.long)

    return src_batch, tgt_batch, src_lengths


# ==================== Subword Tokenization Support ====================

class Multi30kSubwordDataset(Dataset):
    """Dataset for Multi30k with raw text pairs (for subword tokenization)."""

    def __init__(self, hf_dataset, max_seq_length: int = 100):
        self.pairs = []
        for example in hf_dataset:
            src = example['de'].strip()
            tgt = example['en'].strip()
            if src and tgt:
                self.pairs.append((src, tgt))

        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    @property
    def src_texts(self):
        return [p[0] for p in self.pairs]

    @property
    def tgt_texts(self):
        return [p[1] for p in self.pairs]


def collate_subword_batch(
    batch,
    tokenizer: "SubwordTokenizer",
    max_length: Optional[int] = 100
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate batch using subword tokenizer.

    Returns:
        src_batch: (batch, max_src_len) source token IDs
        tgt_batch: (batch, max_tgt_len) target token IDs
        src_lengths: (batch,) actual source lengths
    """
    src_texts = [pair[0] for pair in batch]
    tgt_texts = [pair[1] for pair in batch]

    # Encode with BPE tokenizer (includes <bos> and <eos>)
    src_batch, src_lengths = tokenizer.encode_batch(
        src_texts,
        add_bos=True,
        add_eos=True,
        max_length=max_length,
        padding=True,
        return_tensors=True
    )

    tgt_batch, _ = tokenizer.encode_batch(
        tgt_texts,
        add_bos=True,
        add_eos=True,
        max_length=max_length,
        padding=True,
        return_tensors=True
    )

    return src_batch, tgt_batch, src_lengths
