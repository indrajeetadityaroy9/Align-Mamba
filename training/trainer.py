"""Training loop for NMT with gradient accumulation optimized for H100."""

import os
import itertools
import torch
from torch.amp import autocast
from tqdm import tqdm

from training.evaluator import validate
from training.hardware import GradientAccumulator


def train_model(
    encoder,
    decoder,
    train_data_loader,
    val_data_loader,
    optimizer,
    loss_function,
    target_vocab,
    lr_scheduler,
    num_epochs: int,
    device,
    accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    model_dir: str = 'model'
):
    """
    Train the NMT model with gradient accumulation and BLEU-based checkpointing.

    Optimized for H100 with:
    - BFloat16 mixed precision (native H100 support)
    - Efficient gradient accumulation
    - Non-blocking data transfers

    Args:
        encoder: Encoder model
        decoder: Decoder model
        train_data_loader: Training DataLoader
        val_data_loader: Validation DataLoader
        optimizer: Optimizer
        loss_function: Loss function
        target_vocab: Target vocabulary
        lr_scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Torch device
        accumulation_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        model_dir: Directory to save checkpoints
    """
    best_bleu_score = 0.0
    best_model_filepath = os.path.join(model_dir, 'trained_model.pth')
    os.makedirs(model_dir, exist_ok=True)

    # Use hardware-optimized gradient accumulator
    grad_accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm
    )

    print(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'Val BLEU':>10}")
    print("-" * 40)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0

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

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                encoder_outputs, encoder_hidden = encoder(src_batch, src_lengths)

                decoder_input = tgt_batch[:, 0]
                decoder_hidden = encoder_hidden

                loss = 0
                target_length = tgt_batch.size(1)

                # Teacher forcing
                for t in range(1, target_length):
                    output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    output = output.squeeze(1)
                    loss += loss_function(output, tgt_batch[:, t])
                    decoder_input = tgt_batch[:, t]

                loss = loss / (target_length - 1)

            # Use gradient accumulator for backward and step
            grad_accumulator.backward(loss)
            epoch_loss += loss.item()

            model_params = itertools.chain(encoder.parameters(), decoder.parameters())
            if grad_accumulator.step(model_params):
                pass  # Optimizer step taken

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(train_data_loader)
        lr_scheduler.step()

        # Validation
        bleu_score = validate(val_data_loader, encoder, decoder, target_vocab, device)

        print(f"{epoch + 1:>6} | {avg_epoch_loss:>12.4f} | {bleu_score:>10.2f}")

        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'bleu': bleu_score
            }, best_model_filepath)
            print(f"  -> New best model saved (BLEU: {bleu_score:.2f})")

    print(f"\nTraining complete. Best BLEU: {best_bleu_score:.2f}")
