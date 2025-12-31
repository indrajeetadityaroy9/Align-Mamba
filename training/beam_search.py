"""Beam Search Decoding for NMT.

Beam search explores multiple hypotheses in parallel, typically
improving translation quality by 2-3 BLEU over greedy decoding.

Features:
- Length normalization to avoid favoring short sequences
- Early stopping when all beams are complete
- Support for both standard Decoder and DeepDecoder
"""

from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BeamSearchConfig:
    """Configuration for beam search decoding."""
    beam_size: int = 5
    max_length: int = 100
    length_penalty: float = 0.6  # Alpha in length normalization
    min_length: int = 1
    eos_id: int = 3
    bos_id: int = 2
    pad_id: int = 0
    early_stopping: bool = True


class BeamHypothesis:
    """Container for a single beam hypothesis."""

    def __init__(self, tokens: List[int], score: float, hidden_state: torch.Tensor):
        self.tokens = tokens
        self.score = score  # Log probability
        self.hidden_state = hidden_state

    def __len__(self):
        return len(self.tokens)

    @property
    def normalized_score(self) -> float:
        """Score normalized by length (for final ranking)."""
        return self.score / len(self.tokens)


def length_normalize(score: float, length: int, alpha: float = 0.6) -> float:
    """
    Apply length normalization to beam score.

    Formula: score / ((5 + length) / 6) ^ alpha

    This penalizes very short sequences and normalizes for fair comparison.
    """
    penalty = ((5.0 + length) / 6.0) ** alpha
    return score / penalty


def beam_search_decode(
    encoder,
    decoder,
    src_batch: torch.Tensor,
    src_lengths: torch.Tensor,
    vocab_or_tokenizer,
    device: torch.device,
    config: Optional[BeamSearchConfig] = None
) -> List[List[int]]:
    """
    Beam search decoding for a batch of source sequences.

    Args:
        encoder: Encoder model
        decoder: Decoder model (Decoder or DeepDecoder)
        src_batch: (batch, src_len) source token indices
        src_lengths: (batch,) source lengths
        vocab_or_tokenizer: Vocabulary or SubwordTokenizer for special token IDs
        device: Torch device
        config: Beam search configuration

    Returns:
        List of decoded token sequences (one per batch item)
    """
    if config is None:
        config = BeamSearchConfig()

    encoder.eval()
    decoder.eval()

    # Get special token IDs
    if hasattr(vocab_or_tokenizer, 'bos_id'):
        # SubwordTokenizer
        bos_id = vocab_or_tokenizer.bos_id
        eos_id = vocab_or_tokenizer.eos_id
        pad_id = vocab_or_tokenizer.pad_id
    else:
        # Vocab
        bos_id = vocab_or_tokenizer['<bos>']
        eos_id = vocab_or_tokenizer['<eos>']
        pad_id = vocab_or_tokenizer['<pad>']

    config.bos_id = bos_id
    config.eos_id = eos_id
    config.pad_id = pad_id

    batch_size = src_batch.size(0)
    results = []

    with torch.no_grad():
        # Encode source
        encoder_outputs, encoder_hidden = encoder(src_batch, src_lengths)

        # Decode each sequence individually (batched beam search is complex)
        for b in range(batch_size):
            single_encoder_outputs = encoder_outputs[b:b+1]  # (1, src_len, hidden*2)
            single_encoder_hidden = encoder_hidden[:, b:b+1, :]  # (1, 1, hidden)

            decoded = _beam_search_single(
                decoder=decoder,
                encoder_outputs=single_encoder_outputs,
                encoder_hidden=single_encoder_hidden,
                config=config,
                device=device
            )
            results.append(decoded)

    encoder.train()
    decoder.train()

    return results


def _beam_search_single(
    decoder,
    encoder_outputs: torch.Tensor,
    encoder_hidden: torch.Tensor,
    config: BeamSearchConfig,
    device: torch.device
) -> List[int]:
    """
    Beam search for a single sequence.

    Args:
        decoder: Decoder model
        encoder_outputs: (1, src_len, hidden*2) encoder outputs
        encoder_hidden: (1, 1, hidden) or (num_layers, 1, hidden) encoder hidden
        config: Beam search config
        device: Torch device

    Returns:
        Best decoded token sequence
    """
    beam_size = config.beam_size
    max_length = config.max_length
    eos_id = config.eos_id
    bos_id = config.bos_id

    # Check if this is DeepDecoder (has init_hidden method)
    is_deep_decoder = hasattr(decoder, 'init_hidden')

    # Initialize decoder hidden state
    if is_deep_decoder:
        decoder_hidden = decoder.init_hidden(encoder_hidden)
        # decoder_hidden: (num_layers, 1, hidden)
    else:
        decoder_hidden = encoder_hidden
        # decoder_hidden: (1, 1, hidden)

    # Initialize beams
    # Each beam: (tokens, log_prob, hidden_state, coverage)
    beams = [(
        [bos_id],  # tokens
        0.0,  # log probability
        decoder_hidden,  # hidden state
        None  # coverage (for DeepDecoder)
    )]

    completed = []

    for step in range(max_length):
        all_candidates = []

        for tokens, score, hidden, coverage in beams:
            # Check if this beam is already complete
            if tokens[-1] == eos_id:
                completed.append((tokens, score, hidden, coverage))
                continue

            # Prepare decoder input
            decoder_input = torch.tensor([tokens[-1]], dtype=torch.long, device=device)

            # Forward pass
            if is_deep_decoder:
                output, new_hidden, attn_weights, new_coverage = decoder(
                    decoder_input,
                    hidden,
                    encoder_outputs,
                    coverage=coverage
                )
            else:
                output, new_hidden = decoder(decoder_input, hidden, encoder_outputs)
                new_coverage = None

            # Get log probabilities
            log_probs = F.log_softmax(output.squeeze(1), dim=-1)  # (1, vocab_size)

            # Get top-k candidates
            topk_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            for i in range(beam_size):
                token_id = topk_ids[0, i].item()
                token_log_prob = topk_probs[0, i].item()

                new_tokens = tokens + [token_id]
                new_score = score + token_log_prob

                all_candidates.append((
                    new_tokens,
                    new_score,
                    new_hidden,
                    new_coverage
                ))

        # Early stopping if all beams are complete
        if len(all_candidates) == 0:
            break

        # Select top beams
        # Sort by length-normalized score
        all_candidates.sort(
            key=lambda x: length_normalize(x[1], len(x[0]), config.length_penalty),
            reverse=True
        )
        beams = all_candidates[:beam_size]

        # Check for early stopping
        if config.early_stopping and len(completed) >= beam_size:
            # Check if best completed is better than any active beam
            best_completed = max(
                completed,
                key=lambda x: length_normalize(x[1], len(x[0]), config.length_penalty)
            )
            best_active = beams[0] if beams else None

            if best_active is None or length_normalize(
                best_completed[1], len(best_completed[0]), config.length_penalty
            ) >= length_normalize(
                best_active[1], len(best_active[0]), config.length_penalty
            ):
                break

    # Combine completed and remaining beams
    all_hypotheses = completed + beams

    if not all_hypotheses:
        return [bos_id]

    # Select best hypothesis
    best = max(
        all_hypotheses,
        key=lambda x: length_normalize(x[1], len(x[0]), config.length_penalty)
    )

    return best[0]


def beam_search_validate(
    val_data_loader,
    encoder,
    decoder,
    target_vocab,
    device: torch.device,
    beam_size: int = 5,
    length_penalty: float = 0.6
) -> float:
    """
    Validate model using beam search decoding and compute BLEU.

    This is a drop-in replacement for the greedy validate() function,
    typically improving BLEU by 2-3 points.

    Args:
        val_data_loader: Validation DataLoader
        encoder: Encoder model
        decoder: Decoder model
        target_vocab: Target vocabulary or tokenizer
        device: Torch device
        beam_size: Number of beams
        length_penalty: Length normalization alpha

    Returns:
        BLEU score (0-100)
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    encoder.eval()
    decoder.eval()

    config = BeamSearchConfig(
        beam_size=beam_size,
        length_penalty=length_penalty
    )

    all_references = []
    all_hypotheses = []

    # Check if using BPE tokenizer
    is_subword = hasattr(target_vocab, 'decode') and hasattr(target_vocab, 'bos_id')

    with torch.no_grad():
        for src_batch, tgt_batch, src_lengths in val_data_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            src_lengths = src_lengths.to(device)

            # Beam search decode
            predictions = beam_search_decode(
                encoder=encoder,
                decoder=decoder,
                src_batch=src_batch,
                src_lengths=src_lengths,
                vocab_or_tokenizer=target_vocab,
                device=device,
                config=config
            )

            # Convert to tokens for BLEU
            batch_size = src_batch.size(0)
            for i in range(batch_size):
                ref_indices = tgt_batch[i].cpu().tolist()
                hyp_indices = predictions[i]

                if is_subword:
                    # BPE tokenizer
                    ref_text = target_vocab.decode(ref_indices, skip_special_tokens=True)
                    hyp_text = target_vocab.decode(hyp_indices, skip_special_tokens=True)
                    ref_tokens = ref_text.split() if ref_text else []
                    hyp_tokens = hyp_text.split() if hyp_text else []
                else:
                    # Word-level vocab
                    eos_id = target_vocab['<eos>']
                    special_ids = {target_vocab['<pad>'], target_vocab['<bos>'], eos_id}

                    ref_tokens = []
                    for idx in ref_indices:
                        if idx == eos_id:
                            break
                        if idx not in special_ids:
                            ref_tokens.append(target_vocab.lookup_token(idx))

                    hyp_tokens = []
                    for idx in hyp_indices:
                        if idx == eos_id:
                            break
                        if idx not in special_ids:
                            hyp_tokens.append(target_vocab.lookup_token(idx))

                all_references.append([ref_tokens])
                all_hypotheses.append(hyp_tokens)

    # Compute BLEU
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothing) * 100

    encoder.train()
    decoder.train()

    return bleu_score
