#!/usr/bin/env python3
"""
Build a 32K BPE tokenizer for Document-Level NMT.

CRITICAL: Do NOT use mBART's 250K tokenizer - it makes the model 95% embedding table.
This script creates a proper 32K vocabulary comparable to "Attention Is All You Need".

Usage:
    python scripts/build_tokenizer.py
    python scripts/build_tokenizer.py --vocab_size 16384
    python scripts/build_tokenizer.py --lang_pair de-en
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC


def load_training_data(lang_pair: str = "de-en"):
    """Load training data for tokenizer training."""
    src_lang, tgt_lang = lang_pair.split("-")

    print(f"Loading dataset for {lang_pair}...")

    # Try multiple sources
    texts = []

    # Option 1: Try WMT14 (already cached from dry run)
    try:
        from datasets import load_dataset

        # WMT14 uses different language codes
        wmt_pair = f"{src_lang}-{tgt_lang}"
        dataset = load_dataset("wmt14", wmt_pair, split="train")

        for item in dataset:
            texts.append(item["translation"][src_lang])
            texts.append(item["translation"][tgt_lang])

        print(f"Loaded {len(texts)} sentences from WMT14")
        return texts
    except Exception as e:
        print(f"WMT14 loading failed: {e}")

    # Option 2: Use synthetic data (for testing only)
    print("WARNING: Using synthetic data. Replace with real data for production.")
    for i in range(100000):
        if src_lang == "de":
            texts.append(f"Dies ist ein Testsatz Nummer {i} mit verschiedenen Wörtern.")
        else:
            texts.append(f"Ceci est une phrase de test numéro {i} avec différents mots.")
        texts.append(f"This is a test sentence number {i} with various words.")

    return texts


def build_tokenizer(
    vocab_size: int = 32768,
    lang_pair: str = "de-en",
    output_dir: str = "data/tokenizer",
):
    """
    Build a BPE tokenizer with specified vocabulary size.

    Args:
        vocab_size: Target vocabulary size (default 32768 for scientific comparability)
        lang_pair: Language pair (e.g., "de-en", "fr-en")
        output_dir: Output directory for tokenizer files
    """
    print("=" * 60)
    print(f"Building {vocab_size}-token BPE Tokenizer")
    print(f"Language pair: {lang_pair}")
    print("=" * 60)

    # Load training data
    texts = load_training_data(lang_pair)

    if not texts:
        raise ValueError("No training data loaded!")

    # Create iterator for training
    def batch_iterator(batch_size=1000):
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

    # Initialize BPE Tokenizer with byte-level encoding
    print("\nInitializing BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())

    # Normalization: NFKC unicode normalization
    tokenizer.normalizer = NFKC()

    # Pre-tokenization: Byte-level (handles any unicode)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Decoder: Byte-level to reconstruct original text
    tokenizer.decoder = decoders.ByteLevel()

    # Define special tokens
    special_tokens = [
        "<pad>",   # 0 - Padding
        "<s>",     # 1 - BOS (beginning of sequence)
        "</s>",    # 2 - EOS (end of sequence)
        "<unk>",   # 3 - Unknown
        "<mask>",  # 4 - Mask (for potential MLM pretraining)
    ]

    # Train the tokenizer
    print(f"\nTraining tokenizer (target vocab: {vocab_size})...")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        min_frequency=2,  # Tokens must appear at least twice
    )

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Post-processing: Add BOS/EOS tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # Enable padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<pad>"),
        pad_token="<pad>",
    )

    # Enable truncation
    tokenizer.enable_truncation(max_length=8192)

    # Save the tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_file = output_path / f"tokenizer_{lang_pair}_{vocab_size}.json"
    tokenizer.save(str(tokenizer_file))

    # Also save as default tokenizer.json
    default_file = output_path / "tokenizer.json"
    tokenizer.save(str(default_file))

    # Print summary
    print("\n" + "=" * 60)
    print("Tokenizer Training Complete!")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Saved to: {tokenizer_file}")
    print(f"Default: {default_file}")

    # Test the tokenizer
    print("\n--- Tokenizer Test ---")
    test_sentences = [
        "This is a test sentence.",
        "Dies ist ein Testsatz.",
        "The hybrid Mamba-Attention architecture enables efficient document-level translation.",
    ]

    for sent in test_sentences:
        encoded = tokenizer.encode(sent)
        print(f"\nInput: {sent}")
        print(f"Tokens: {encoded.tokens[:20]}...")
        print(f"IDs: {encoded.ids[:20]}...")
        print(f"Length: {len(encoded.ids)}")

    # Print special token IDs
    print("\n--- Special Token IDs ---")
    for token in special_tokens:
        print(f"{token}: {tokenizer.token_to_id(token)}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Build BPE tokenizer for NMT")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32768,
        help="Vocabulary size (default: 32768)",
    )
    parser.add_argument(
        "--lang_pair",
        type=str,
        default="de-en",
        help="Language pair (default: de-en)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tokenizer",
        help="Output directory (default: data/tokenizer)",
    )

    args = parser.parse_args()

    build_tokenizer(
        vocab_size=args.vocab_size,
        lang_pair=args.lang_pair,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
