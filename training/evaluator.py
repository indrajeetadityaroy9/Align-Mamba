"""Validation and BLEU evaluation for NMT."""

from typing import Union

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def _is_subword_tokenizer(vocab_or_tokenizer) -> bool:
    """Check if the vocab/tokenizer is a SubwordTokenizer."""
    return hasattr(vocab_or_tokenizer, 'decode') and hasattr(vocab_or_tokenizer, 'bos_id')


def _get_bos_id(vocab_or_tokenizer) -> int:
    """Get BOS token ID from vocab or tokenizer."""
    if _is_subword_tokenizer(vocab_or_tokenizer):
        return vocab_or_tokenizer.bos_id
    return vocab_or_tokenizer['<bos>']


def _get_eos_id(vocab_or_tokenizer) -> int:
    """Get EOS token ID from vocab or tokenizer."""
    if _is_subword_tokenizer(vocab_or_tokenizer):
        return vocab_or_tokenizer.eos_id
    return vocab_or_tokenizer['<eos>']


def _decode_tokens(indices: list, vocab_or_tokenizer, skip_special: bool = True) -> list:
    """
    Decode token IDs to word tokens for BLEU scoring.

    For BPE tokenizer: decode to string, then split into words
    For word vocab: lookup each token individually
    """
    if _is_subword_tokenizer(vocab_or_tokenizer):
        # BPE tokenizer: decode to string, split into words
        text = vocab_or_tokenizer.decode(indices, skip_special_tokens=skip_special)
        return text.split() if text else []
    else:
        # Word-level vocab: lookup tokens
        tokens = []
        eos_id = vocab_or_tokenizer['<eos>']
        special_ids = {vocab_or_tokenizer['<pad>'], vocab_or_tokenizer['<bos>'], eos_id}

        for idx in indices:
            if idx == eos_id:
                break
            if idx in special_ids:
                continue
            tokens.append(vocab_or_tokenizer.lookup_token(idx))
        return tokens


def validate(val_data_loader, encoder, decoder, target_vocab, device) -> float:
    """
    Validate model and compute corpus BLEU score.

    Args:
        val_data_loader: DataLoader for validation data
        encoder: Encoder model
        decoder: Decoder model
        target_vocab: Target vocabulary (Vocab) or SubwordTokenizer
        device: Torch device

    Returns:
        BLEU score (0-100)
    """
    encoder.eval()
    decoder.eval()

    all_references = []
    all_hypotheses = []

    bos_id = _get_bos_id(target_vocab)
    eos_id = _get_eos_id(target_vocab)

    with torch.no_grad():
        for src_batch, tgt_batch, src_lengths in val_data_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            src_lengths = src_lengths.to(device)

            encoder_outputs, encoder_hidden = encoder(src_batch, src_lengths)

            batch_size = src_batch.size(0)
            decoder_input = torch.full(
                (batch_size,), bos_id,
                dtype=torch.long, device=device
            )
            decoder_hidden = encoder_hidden

            predictions = []
            max_len = tgt_batch.size(1)

            for _ in range(max_len):
                output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = output.argmax(dim=-1).squeeze(1)
                predictions.append(decoder_input)

            predictions = torch.stack(predictions, dim=1)

            # Convert to tokens for BLEU
            for i in range(batch_size):
                ref_indices = tgt_batch[i].cpu().tolist()
                hyp_indices = predictions[i].cpu().tolist()

                # Decode reference (skip special tokens)
                ref_tokens = _decode_tokens(ref_indices, target_vocab, skip_special=True)

                # Decode hypothesis (stops at EOS)
                hyp_tokens = _decode_tokens(hyp_indices, target_vocab, skip_special=True)

                all_references.append([ref_tokens])
                all_hypotheses.append(hyp_tokens)

    # Compute corpus BLEU with smoothing
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothing) * 100

    encoder.train()
    decoder.train()

    return bleu_score
