"""
ORPO Training Loop for NMT.

Implements preference optimization with self-generated rejected samples.
"""

import os
import itertools
import torch
from torch.amp import autocast
from tqdm import tqdm

from training.evaluator import validate
from training.orpo_loss import ORPOLoss


@torch.no_grad()
def generate_rejected(
    encoder,
    decoder,
    src_batch: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int,
    bos_idx: int,
    eos_idx: int
) -> torch.Tensor:
    """
    Generate rejected translations via greedy decoding.

    Args:
        encoder: Encoder model
        decoder: Decoder model
        src_batch: (batch, src_len) source tokens
        src_lengths: (batch,) source lengths
        max_len: Maximum generation length
        bos_idx: Beginning of sequence token index
        eos_idx: End of sequence token index

    Returns:
        rejected: (batch, max_len) generated token indices
    """
    encoder.eval()
    decoder.eval()

    encoder_outputs, encoder_hidden = encoder(src_batch, src_lengths)

    batch_size = src_batch.size(0)
    device = src_batch.device

    decoder_input = torch.full((batch_size,), bos_idx, dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden

    rejected_tokens = [decoder_input]

    for _ in range(max_len - 1):
        output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_input = output.argmax(dim=-1).squeeze(1)
        rejected_tokens.append(decoder_input)

    rejected = torch.stack(rejected_tokens, dim=1)

    encoder.train()
    decoder.train()

    return rejected


def decode_sequence(
    decoder,
    encoder_outputs: torch.Tensor,
    decoder_hidden: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Run decoder with teacher forcing, return all logits.

    Args:
        decoder: Decoder model
        encoder_outputs: (batch, src_len, hidden*2) encoder outputs
        decoder_hidden: (1, batch, hidden) initial hidden state
        target: (batch, tgt_len) target tokens for teacher forcing

    Returns:
        logits: (batch, tgt_len-1, vocab_size) output logits
    """
    tgt_len = target.size(1)
    all_logits = []

    decoder_input = target[:, 0]  # Start with <bos>

    for t in range(1, tgt_len):
        output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        all_logits.append(output.squeeze(1))
        decoder_input = target[:, t]  # Teacher forcing

    return torch.stack(all_logits, dim=1)


def train_orpo(
    encoder,
    decoder,
    train_data_loader,
    val_data_loader,
    optimizer,
    target_vocab,
    lr_scheduler,
    num_epochs: int,
    device,
    beta: float = 0.1,
    accumulation_steps: int = 8,
    model_dir: str = 'model'
):
    """
    ORPO fine-tuning loop.

    Args:
        encoder: Encoder model
        decoder: Decoder model
        train_data_loader: Training DataLoader
        val_data_loader: Validation DataLoader
        optimizer: Optimizer
        target_vocab: Target vocabulary
        lr_scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Torch device
        beta: ORPO loss weight (default: 0.1)
        accumulation_steps: Gradient accumulation steps
        model_dir: Directory to save checkpoints
    """
    orpo_loss_fn = ORPOLoss(beta=beta, pad_idx=target_vocab['<pad>'])

    best_bleu_score = 0.0
    best_model_filepath = os.path.join(model_dir, 'orpo_model.pth')
    os.makedirs(model_dir, exist_ok=True)

    bos_idx = target_vocab['<bos>']
    eos_idx = target_vocab['<eos>']

    print(f"\n{'Epoch':>6} | {'NLL':>8} | {'ORPO':>8} | {'LogOdds':>8} | {'BLEU':>8}")
    print("-" * 55)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        epoch_nll = 0.0
        epoch_orpo = 0.0
        epoch_log_odds = 0.0

        progress_bar = tqdm(
            enumerate(train_data_loader, 1),
            total=len(train_data_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False
        )

        for batch_idx, (src_batch, tgt_batch, src_lengths) in progress_bar:
            # Non-blocking transfers for H100 PCIe/NVLink optimization
            src_batch = src_batch.to(device, non_blocking=True)
            tgt_batch = tgt_batch.to(device, non_blocking=True)
            src_lengths = src_lengths.to(device, non_blocking=True)

            # Chosen = ground truth
            chosen = tgt_batch

            # Generate rejected samples (greedy decode)
            rejected = generate_rejected(
                encoder, decoder,
                src_batch, src_lengths,
                max_len=tgt_batch.size(1),
                bos_idx=bos_idx,
                eos_idx=eos_idx
            )

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Encode source (once)
                encoder_outputs, encoder_hidden = encoder(src_batch, src_lengths)

                # Decode chosen (teacher forcing)
                chosen_logits = decode_sequence(
                    decoder, encoder_outputs, encoder_hidden.clone(), chosen
                )

                # Decode rejected (teacher forcing on generated sequence)
                rejected_logits = decode_sequence(
                    decoder, encoder_outputs, encoder_hidden.clone(), rejected
                )

                # ORPO loss (shift labels by 1 to align with logits)
                loss, metrics = orpo_loss_fn(
                    chosen_logits, chosen[:, 1:],
                    rejected_logits, rejected[:, 1:]
                )

                loss = loss / accumulation_steps

            loss.backward()

            epoch_nll += metrics['nll']
            epoch_orpo += metrics['orpo']
            epoch_log_odds += metrics['log_odds']

            if batch_idx % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(encoder.parameters(), decoder.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({
                'nll': f"{metrics['nll']:.3f}",
                'orpo': f"{metrics['orpo']:.3f}",
                'log_odds': f"{metrics['log_odds']:.2f}"
            })

        # Epoch averages
        n_batches = len(train_data_loader)
        avg_nll = epoch_nll / n_batches
        avg_orpo = epoch_orpo / n_batches
        avg_log_odds = epoch_log_odds / n_batches

        lr_scheduler.step()

        # Validation
        bleu_score = validate(val_data_loader, encoder, decoder, target_vocab, device)

        print(f"{epoch + 1:>6} | {avg_nll:>8.4f} | {avg_orpo:>8.4f} | {avg_log_odds:>8.3f} | {bleu_score:>8.2f}")

        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'bleu': bleu_score,
                'beta': beta
            }, best_model_filepath)
            print(f"  -> New best ORPO model saved (BLEU: {bleu_score:.2f})")

    print(f"\nORPO training complete. Best BLEU: {best_bleu_score:.2f}")
