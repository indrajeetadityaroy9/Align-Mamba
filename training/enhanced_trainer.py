"""Enhanced Training Loop for PhD-Level NMT.

Integrates all advanced training techniques:
- Label smoothing for better calibration
- Warmup + cosine LR decay for stable Transformer training
- R-Drop consistency regularization
- Coverage loss for attention regularization
- Beam search validation for accurate BLEU
- Gradient accumulation for effective larger batches
"""

import os
import itertools
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from training.evaluator import validate
from training.beam_search import beam_search_validate, BeamSearchConfig
from training.label_smoothing import LabelSmoothingCrossEntropy
from training.lr_scheduler import WarmupCosineScheduler
from training.rdrop import compute_kl_divergence
from training.hardware import GradientAccumulator
from models.decoder import compute_coverage_loss


def train_enhanced(
    encoder,
    decoder,
    train_data_loader,
    val_data_loader,
    optimizer,
    target_vocab,
    num_epochs: int,
    device,
    # Training enhancements
    label_smoothing: float = 0.1,
    warmup_epochs: float = 1.5,
    use_rdrop: bool = True,
    rdrop_alpha: float = 0.7,
    use_coverage: bool = True,
    coverage_lambda: float = 0.5,
    # Optimization
    accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    peak_lr: float = 0.0003,
    min_lr: float = 1e-7,
    # Validation
    use_beam_search: bool = True,
    beam_size: int = 5,
    length_penalty: float = 0.6,
    # Paths
    model_dir: str = 'model'
):
    """
    Enhanced training loop with all PhD-level improvements.

    Args:
        encoder: Encoder model (BiGRU, Mamba, or Transformer)
        decoder: Decoder model (Decoder or DeepDecoder)
        train_data_loader: Training DataLoader
        val_data_loader: Validation DataLoader
        optimizer: Optimizer (AdamW recommended)
        target_vocab: Target vocabulary or SubwordTokenizer
        num_epochs: Number of training epochs
        device: Torch device

        # Training enhancements
        label_smoothing: Label smoothing epsilon (0.1 typical)
        warmup_epochs: LR warmup epochs (1.5 typical for Transformer)
        use_rdrop: Enable R-Drop consistency regularization
        rdrop_alpha: R-Drop KL weight (0.7 typical)
        use_coverage: Enable coverage loss (only with DeepDecoder)
        coverage_lambda: Coverage loss weight (0.5 typical)

        # Optimization
        accumulation_steps: Gradient accumulation steps
        max_grad_norm: Gradient clipping norm
        peak_lr: Peak learning rate after warmup
        min_lr: Minimum learning rate at end

        # Validation
        use_beam_search: Use beam search for validation BLEU
        beam_size: Beam search width
        length_penalty: Beam search length penalty

        # Paths
        model_dir: Directory for checkpoints
    """
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'trained_model.pth')
    best_bleu = 0.0

    # Get vocab size and pad index
    if hasattr(target_vocab, 'vocab_size'):
        vocab_size = target_vocab.vocab_size
        pad_idx = target_vocab.pad_id
    else:
        vocab_size = len(target_vocab)
        pad_idx = target_vocab['<pad>']

    # Check if using DeepDecoder (supports coverage)
    is_deep_decoder = hasattr(decoder, 'use_coverage')
    use_coverage = use_coverage and is_deep_decoder

    # Loss function with label smoothing
    loss_fn = LabelSmoothingCrossEntropy(
        smoothing=label_smoothing,
        padding_idx=pad_idx
    )

    # LR scheduler with warmup + cosine decay
    steps_per_epoch = len(train_data_loader) // accumulation_steps
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        peak_lr=peak_lr,
        min_lr=min_lr
    )

    # Gradient accumulator
    grad_accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm
    )

    # Print configuration
    print("\n" + "=" * 60)
    print("Enhanced NMT Training Configuration")
    print("=" * 60)
    print(f"Encoder type: {encoder.__class__.__name__}")
    print(f"Decoder type: {decoder.__class__.__name__}")
    print(f"Vocab size: {vocab_size}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Peak LR: {peak_lr}")
    print(f"R-Drop: {use_rdrop} (alpha={rdrop_alpha})")
    print(f"Coverage loss: {use_coverage} (lambda={coverage_lambda})")
    print(f"Beam search validation: {use_beam_search} (size={beam_size})")
    print(f"Gradient accumulation: {accumulation_steps} steps")
    print("=" * 60 + "\n")

    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val BLEU':>10} | {'LR':>10}")
    print("-" * 50)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_cov_loss = 0.0

        progress = tqdm(
            enumerate(train_data_loader, 1),
            total=len(train_data_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False
        )

        for batch_idx, (src_batch, tgt_batch, src_lengths) in progress:
            src_batch = src_batch.to(device, non_blocking=True)
            tgt_batch = tgt_batch.to(device, non_blocking=True)
            src_lengths = src_lengths.to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Forward pass 1
                logits1, cov_loss1 = _forward_pass(
                    encoder, decoder, src_batch, tgt_batch, src_lengths,
                    is_deep_decoder, use_coverage
                )

                # Cross-entropy loss
                ce_loss = loss_fn(logits1, tgt_batch[:, 1:])

                if use_rdrop:
                    # Forward pass 2 (different dropout mask)
                    logits2, cov_loss2 = _forward_pass(
                        encoder, decoder, src_batch, tgt_batch, src_lengths,
                        is_deep_decoder, use_coverage
                    )

                    # Second CE loss
                    ce_loss2 = loss_fn(logits2, tgt_batch[:, 1:])
                    ce_loss = 0.5 * (ce_loss + ce_loss2)

                    # KL divergence loss
                    padding_mask = tgt_batch[:, 1:].eq(pad_idx)
                    kl_loss = compute_kl_divergence(logits1, logits2, padding_mask)
                    loss = ce_loss + rdrop_alpha * kl_loss

                    # Coverage loss (average of both passes)
                    if use_coverage:
                        cov_loss = 0.5 * (cov_loss1 + cov_loss2)
                        loss = loss + coverage_lambda * cov_loss
                else:
                    kl_loss = torch.tensor(0.0, device=device)
                    loss = ce_loss

                    if use_coverage:
                        loss = loss + coverage_lambda * cov_loss1
                        cov_loss = cov_loss1
                    else:
                        cov_loss = torch.tensor(0.0, device=device)

            # Backward with gradient accumulation
            grad_accumulator.backward(loss)

            model_params = itertools.chain(encoder.parameters(), decoder.parameters())
            if grad_accumulator.step(model_params):
                scheduler.step()

            # Track losses
            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss.item()
            if use_rdrop:
                epoch_kl_loss += kl_loss.item()
            if use_coverage:
                epoch_cov_loss += cov_loss.item()

            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # Epoch statistics
        n_batches = len(train_data_loader)
        avg_loss = epoch_loss / n_batches
        avg_ce = epoch_ce_loss / n_batches
        avg_kl = epoch_kl_loss / n_batches if use_rdrop else 0
        avg_cov = epoch_cov_loss / n_batches if use_coverage else 0

        # Validation
        if use_beam_search:
            bleu = beam_search_validate(
                val_data_loader, encoder, decoder, target_vocab, device,
                beam_size=beam_size, length_penalty=length_penalty
            )
        else:
            bleu = validate(val_data_loader, encoder, decoder, target_vocab, device)

        current_lr = scheduler.get_last_lr()[0]
        print(f"{epoch+1:>6} | {avg_loss:>12.4f} | {bleu:>10.2f} | {current_lr:>10.2e}")

        # Log component losses
        if use_rdrop or use_coverage:
            components = []
            components.append(f"CE={avg_ce:.4f}")
            if use_rdrop:
                components.append(f"KL={avg_kl:.4f}")
            if use_coverage:
                components.append(f"Cov={avg_cov:.4f}")
            print(f"         Components: {', '.join(components)}")

        # Save best model
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'bleu': bleu,
                'epoch': epoch + 1,
                'config': {
                    'encoder_type': encoder.__class__.__name__,
                    'decoder_type': decoder.__class__.__name__,
                    'label_smoothing': label_smoothing,
                    'use_rdrop': use_rdrop,
                    'use_coverage': use_coverage,
                }
            }, best_model_path)
            print(f"  -> New best model saved (BLEU: {bleu:.2f})")

    print(f"\nTraining complete. Best BLEU: {best_bleu:.2f}")
    return best_bleu


def _forward_pass(
    encoder,
    decoder,
    src_batch: torch.Tensor,
    tgt_batch: torch.Tensor,
    src_lengths: torch.Tensor,
    is_deep_decoder: bool,
    use_coverage: bool
):
    """
    Single forward pass through encoder-decoder.

    Returns:
        logits: (batch, tgt_len-1, vocab_size) output logits
        coverage_loss: Scalar coverage loss (0 if not using coverage)
    """
    encoder_outputs, encoder_hidden = encoder(src_batch, src_lengths)

    batch_size = src_batch.size(0)
    tgt_len = tgt_batch.size(1)

    # Initialize decoder
    decoder_input = tgt_batch[:, 0]

    if is_deep_decoder:
        decoder_hidden = decoder.init_hidden(encoder_hidden)
        coverage = None
        all_coverage_losses = []
    else:
        decoder_hidden = encoder_hidden

    all_logits = []

    # Teacher forcing
    for t in range(1, tgt_len):
        if is_deep_decoder:
            output, decoder_hidden, attn_weights, coverage = decoder(
                decoder_input, decoder_hidden, encoder_outputs, coverage=coverage
            )
            if use_coverage and t > 1:
                # Compute coverage loss (skip first step)
                cov_loss = compute_coverage_loss(attn_weights, coverage - attn_weights)
                all_coverage_losses.append(cov_loss)
        else:
            output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        all_logits.append(output.squeeze(1))
        decoder_input = tgt_batch[:, t]

    logits = torch.stack(all_logits, dim=1)

    # Average coverage loss
    if is_deep_decoder and use_coverage and all_coverage_losses:
        coverage_loss = sum(all_coverage_losses) / len(all_coverage_losses)
    else:
        coverage_loss = torch.tensor(0.0, device=src_batch.device)

    return logits, coverage_loss
