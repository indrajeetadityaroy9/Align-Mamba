"""SentencePiece BPE Subword Tokenizer for NMT.

This module provides a joint BPE tokenizer for German-English translation,
significantly reducing OOV rate and vocabulary size compared to word-level tokenization.
"""

import os
import tempfile
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import sentencepiece as spm


@dataclass
class TokenizerConfig:
    """Configuration for subword tokenizer."""
    vocab_size: int = 16000
    model_type: str = 'bpe'  # 'bpe' or 'unigram'
    character_coverage: float = 1.0
    model_prefix: str = 'model/spm'
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3


class SubwordTokenizer:
    """
    SentencePiece-based subword tokenizer.

    Supports:
    - BPE (byte pair encoding) or Unigram language model
    - Joint source-target vocabulary
    - Special tokens: <pad>, <unk>, <bos>, <eos>

    Usage:
        # Training
        tokenizer = SubwordTokenizer(config)
        tokenizer.train(german_texts + english_texts)

        # Encoding
        ids = tokenizer.encode("Hello world")  # Returns [2, 1234, 5678, 3]

        # Decoding
        text = tokenizer.decode([2, 1234, 5678, 3])  # Returns "Hello world"
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.sp_model: Optional[spm.SentencePieceProcessor] = None
        self._trained = False

    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        Train SentencePiece model from texts.

        Args:
            texts: List of training sentences (should include both src and tgt)
            verbose: Print training progress
        """
        # Create model directory if needed
        model_dir = os.path.dirname(self.config.model_prefix)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        # Write texts to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for text in texts:
                text = text.strip()
                if text:
                    f.write(text + '\n')
            temp_path = f.name

        try:
            # SentencePiece training arguments
            train_args = [
                f'--input={temp_path}',
                f'--model_prefix={self.config.model_prefix}',
                f'--vocab_size={self.config.vocab_size}',
                f'--model_type={self.config.model_type}',
                f'--character_coverage={self.config.character_coverage}',
                f'--pad_id={self.config.pad_id}',
                f'--unk_id={self.config.unk_id}',
                f'--bos_id={self.config.bos_id}',
                f'--eos_id={self.config.eos_id}',
                '--pad_piece=<pad>',
                '--unk_piece=<unk>',
                '--bos_piece=<bos>',
                '--eos_piece=<eos>',
                '--normalization_rule_name=nfkc',
                '--shuffle_input_sentence=true',
                '--input_sentence_size=1000000',
                '--train_extremely_large_corpus=false',
            ]

            if verbose:
                print(f"Training SentencePiece model with {len(texts)} sentences...")
                print(f"  Vocab size: {self.config.vocab_size}")
                print(f"  Model type: {self.config.model_type}")

            spm.SentencePieceTrainer.Train(' '.join(train_args))

            if verbose:
                print(f"Model saved to: {self.config.model_prefix}.model")

        finally:
            # Clean up temp file
            os.unlink(temp_path)

        # Load the trained model
        self.load()
        self._trained = True

    def load(self, model_path: Optional[str] = None) -> None:
        """
        Load trained SentencePiece model.

        Args:
            model_path: Path to .model file (uses config prefix if None)
        """
        if model_path is None:
            model_path = f'{self.config.model_prefix}.model'

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_path)
        self._trained = True

    def save(self, path: Optional[str] = None) -> None:
        """Save tokenizer config (model is already saved during training)."""
        if path is None:
            path = f'{self.config.model_prefix}_config.pt'

        torch.save({
            'vocab_size': self.config.vocab_size,
            'model_type': self.config.model_type,
            'model_prefix': self.config.model_prefix,
        }, path)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to subword token IDs.

        Args:
            text: Input text string
            add_bos: Add <bos> token at start
            add_eos: Add <eos> token at end
            max_length: Maximum sequence length (truncates if exceeded)

        Returns:
            List of token IDs
        """
        if not self._trained or self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")

        # Encode text (without special tokens)
        ids = self.sp_model.EncodeAsIds(text.strip().lower())

        # Handle max_length (reserve space for special tokens)
        if max_length is not None:
            reserved = int(add_bos) + int(add_eos)
            ids = ids[:max_length - reserved]

        # Add special tokens
        if add_bos:
            ids = [self.config.bos_id] + ids
        if add_eos:
            ids = ids + [self.config.eos_id]

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode batch of texts with padding.

        Args:
            texts: List of input texts
            add_bos: Add <bos> token
            add_eos: Add <eos> token
            max_length: Maximum sequence length
            padding: Pad to max length in batch
            return_tensors: Return PyTorch tensors

        Returns:
            input_ids: (batch, seq_len) token IDs
            lengths: (batch,) actual lengths before padding
        """
        encoded = [self.encode(text, add_bos, add_eos, max_length) for text in texts]
        lengths = [len(seq) for seq in encoded]

        if padding:
            max_len = max(lengths)
            padded = [seq + [self.config.pad_id] * (max_len - len(seq)) for seq in encoded]
        else:
            padded = encoded

        if return_tensors:
            return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
        return padded, lengths

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Remove <pad>, <bos>, <eos> from output

        Returns:
            Decoded text string
        """
        if not self._trained or self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")

        if skip_special_tokens:
            special_ids = {self.config.pad_id, self.config.bos_id, self.config.eos_id}
            ids = [i for i in ids if i not in special_ids]

        return self.sp_model.DecodeIds(ids)

    def decode_batch(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of token ID sequences."""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces (for debugging/analysis)."""
        if not self._trained or self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")
        return self.sp_model.EncodeAsPieces(text.strip().lower())

    @property
    def vocab_size(self) -> int:
        """Return actual vocabulary size."""
        if self.sp_model is None:
            return self.config.vocab_size
        return self.sp_model.GetPieceSize()

    @property
    def pad_id(self) -> int:
        return self.config.pad_id

    @property
    def unk_id(self) -> int:
        return self.config.unk_id

    @property
    def bos_id(self) -> int:
        return self.config.bos_id

    @property
    def eos_id(self) -> int:
        return self.config.eos_id

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, token: str) -> int:
        """Get token ID for a token string."""
        if not self._trained or self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")
        return self.sp_model.PieceToId(token)

    def id_to_piece(self, id: int) -> str:
        """Get token string for a token ID."""
        if not self._trained or self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")
        return self.sp_model.IdToPiece(id)


def create_joint_tokenizer(
    src_texts: List[str],
    tgt_texts: List[str],
    vocab_size: int = 16000,
    model_prefix: str = 'model/spm',
    verbose: bool = True
) -> SubwordTokenizer:
    """
    Create and train a joint source-target BPE tokenizer.

    Joint vocabulary is preferred for related language pairs
    as it enables better parameter sharing and handles shared subwords.

    Args:
        src_texts: Source language texts (German)
        tgt_texts: Target language texts (English)
        vocab_size: Target vocabulary size
        model_prefix: Where to save the model
        verbose: Print progress

    Returns:
        Trained SubwordTokenizer
    """
    config = TokenizerConfig(
        vocab_size=vocab_size,
        model_type='bpe',
        model_prefix=model_prefix
    )

    tokenizer = SubwordTokenizer(config)

    # Combine all texts for joint vocabulary
    all_texts = src_texts + tgt_texts

    if verbose:
        print(f"Creating joint tokenizer from {len(src_texts)} src + {len(tgt_texts)} tgt texts")

    tokenizer.train(all_texts, verbose=verbose)

    return tokenizer


def load_tokenizer(model_prefix: str = 'model/spm') -> SubwordTokenizer:
    """
    Load a pre-trained tokenizer.

    Args:
        model_prefix: Path prefix for the model files

    Returns:
        Loaded SubwordTokenizer
    """
    config = TokenizerConfig(model_prefix=model_prefix)
    tokenizer = SubwordTokenizer(config)
    tokenizer.load()
    return tokenizer
