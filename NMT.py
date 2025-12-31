"""
NMT CLI: Multi30k German-to-English Translation

Supports BiGRU, Mamba-2 SSM, and Transformer encoders for architecture comparison.
Supports both word-level and BPE subword tokenization.
"""

import os
import sys
import argparse
import itertools
from functools import partial

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split

from models import create_encoder, EncoderConfig, Decoder, DeepDecoder
from data import (
    load_multi30k, Multi30kDataset, Multi30kSubwordDataset,
    build_vocabulary, collate_batch, collate_subword_batch,
    create_joint_tokenizer, load_tokenizer
)
from training import (
    train_model, validate, train_orpo, train_enhanced,
    configure_hardware, create_optimized_dataloader,
    beam_search_validate
)


# Hyperparameters
MAX_SEQ_LENGTH = 100
EMBEDDING_DIM = 512
HIDDEN_SIZE = 1024


def main():
    parser = argparse.ArgumentParser(
        description='NMT: Multi30k German->English Translation',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument('--encoder-type', choices=['bigru', 'mamba2', 'transformer'], default='transformer')
    train_parser.add_argument('--tokenizer', choices=['word', 'bpe'], default='bpe',
                              help="Tokenization: 'word' (legacy) or 'bpe' (subword)")
    train_parser.add_argument('--vocab-size', type=int, default=16000,
                              help="BPE vocabulary size (only for --tokenizer bpe)")
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--lr', type=float, default=0.0003,
                              help="Learning rate (0.0003 for Transformer, 0.001 for RNN)")

    # Test command
    test_parser = subparsers.add_parser('test', help="Test the model")
    test_parser.add_argument('--encoder-type', choices=['bigru', 'mamba2', 'transformer'], default='transformer')
    test_parser.add_argument('--tokenizer', choices=['word', 'bpe'], default='bpe')

    # ORPO fine-tuning command
    orpo_parser = subparsers.add_parser('train-orpo', help="Fine-tune with ORPO")
    orpo_parser.add_argument('--encoder-type', choices=['bigru', 'mamba2', 'transformer'], default='transformer')
    orpo_parser.add_argument('--tokenizer', choices=['word', 'bpe'], default='bpe')
    orpo_parser.add_argument('--epochs', type=int, default=5)
    orpo_parser.add_argument('--batch-size', type=int, default=32)
    orpo_parser.add_argument('--lr', type=float, default=5e-5)
    orpo_parser.add_argument('--beta', type=float, default=0.1, help="ORPO loss weight")
    orpo_parser.add_argument('--checkpoint', type=str, default='model/trained_model.pth')

    # Enhanced training command (all PhD-level improvements)
    enhanced_parser = subparsers.add_parser('train-enhanced', help="Train with all PhD-level enhancements")
    enhanced_parser.add_argument('--encoder-type', choices=['bigru', 'mamba2', 'transformer'], default='transformer')
    enhanced_parser.add_argument('--decoder-type', choices=['standard', 'deep'], default='deep',
                                  help="Decoder: 'standard' (baseline) or 'deep' (multi-head attention + coverage)")
    enhanced_parser.add_argument('--tokenizer', choices=['word', 'bpe'], default='bpe')
    enhanced_parser.add_argument('--vocab-size', type=int, default=16000)
    enhanced_parser.add_argument('--epochs', type=int, default=15)
    enhanced_parser.add_argument('--batch-size', type=int, default=64)
    enhanced_parser.add_argument('--lr', type=float, default=0.0003)
    enhanced_parser.add_argument('--warmup-epochs', type=float, default=1.5)
    enhanced_parser.add_argument('--label-smoothing', type=float, default=0.1)
    enhanced_parser.add_argument('--rdrop-alpha', type=float, default=0.7)
    enhanced_parser.add_argument('--no-rdrop', action='store_true', help="Disable R-Drop")
    enhanced_parser.add_argument('--coverage-lambda', type=float, default=0.5)
    enhanced_parser.add_argument('--no-coverage', action='store_true', help="Disable coverage loss")
    enhanced_parser.add_argument('--beam-size', type=int, default=5)
    enhanced_parser.add_argument('--length-penalty', type=float, default=0.6)
    enhanced_parser.add_argument('--no-beam-search', action='store_true', help="Use greedy decoding for validation")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Configure hardware for optimal H100 performance
    device = configure_hardware(verbose=True)

    src_lang, tgt_lang = 'german', 'english'
    src_vocab_path, tgt_vocab_path = 'model/src_vocab.pth', 'model/tgt_vocab.pth'

    if args.command == 'train':
        batch_size = args.batch_size
        num_epochs = args.epochs
        learning_rate = args.lr
        use_bpe = args.tokenizer == 'bpe'

        # Load dataset
        hf_dataset = load_multi30k()

        if use_bpe:
            # BPE subword tokenization
            full_dataset = Multi30kSubwordDataset(hf_dataset['train'], max_seq_length=MAX_SEQ_LENGTH)
            print(f"Training pairs: {len(full_dataset)}")

            train_size = int(0.95 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Train or load BPE tokenizer
            tokenizer_path = 'model/spm.model'
            os.makedirs('model', exist_ok=True)

            if os.path.exists(tokenizer_path):
                print(f"Loading existing BPE tokenizer from {tokenizer_path}")
                tokenizer = load_tokenizer('model/spm')
            else:
                print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
                tokenizer = create_joint_tokenizer(
                    full_dataset.src_texts,
                    full_dataset.tgt_texts,
                    vocab_size=args.vocab_size,
                    model_prefix='model/spm',
                    verbose=True
                )

            vocab_size = tokenizer.vocab_size
            pad_idx = tokenizer.pad_id
            print(f"BPE vocab size: {vocab_size}")

            collate_fn = partial(collate_subword_batch, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)
        else:
            # Legacy word-level tokenization
            full_dataset = Multi30kDataset(hf_dataset['train'], max_seq_length=MAX_SEQ_LENGTH)
            print(f"Training pairs: {len(full_dataset)}")

            train_size = int(0.95 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Build vocabularies
            src_vocab = build_vocabulary(full_dataset.src_sequences, src_lang)
            tgt_vocab = build_vocabulary(full_dataset.tgt_sequences, tgt_lang)
            print(f"Vocab sizes - src: {len(src_vocab)}, tgt: {len(tgt_vocab)}")

            vocab_size = len(src_vocab)  # For encoder
            tgt_vocab_size = len(tgt_vocab)  # For decoder
            pad_idx = src_vocab['<pad>']

            collate_fn = partial(collate_batch, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                                src_lang=src_lang, tgt_lang=tgt_lang)

        train_loader = create_optimized_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=True
        )
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, drop_last=False
        )

        # Initialize models
        print(f"\nEncoder: {args.encoder_type}, Tokenizer: {args.tokenizer}")
        encoder = create_encoder(EncoderConfig(
            encoder_type=args.encoder_type,
            vocab_size=vocab_size if use_bpe else len(src_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_SIZE,
            padding_idx=pad_idx
        )).to(device)

        decoder_vocab_size = vocab_size if use_bpe else len(tgt_vocab)
        decoder_pad_idx = pad_idx if use_bpe else tgt_vocab['<pad>']
        decoder = Decoder(HIDDEN_SIZE, decoder_vocab_size, EMBEDDING_DIM, decoder_pad_idx).to(device)

        enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
        dec_params = sum(p.numel() for p in decoder.parameters()) / 1e6
        print(f"Params - encoder: {enc_params:.2f}M, decoder: {dec_params:.2f}M")

        optimizer = torch.optim.Adam(
            itertools.chain(encoder.parameters(), decoder.parameters()), lr=learning_rate
        )
        lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx if use_bpe else tgt_vocab['<pad>'])

        # Pass tokenizer or vocab for decoding during validation
        if use_bpe:
            train_model(encoder, decoder, train_loader, val_loader, optimizer,
                        loss_fn, tokenizer, lr_scheduler, num_epochs, device)
        else:
            train_model(encoder, decoder, train_loader, val_loader, optimizer,
                        loss_fn, tgt_vocab, lr_scheduler, num_epochs, device)

            # Save word-level vocabularies
            os.makedirs('model', exist_ok=True)
            torch.save(src_vocab, src_vocab_path)
            torch.save(tgt_vocab, tgt_vocab_path)
            print("Vocabularies saved.")

    elif args.command == 'test':
        use_bpe = args.tokenizer == 'bpe'
        tokenizer_path = 'model/spm.model'

        if use_bpe:
            if not os.path.exists(tokenizer_path):
                print("Error: BPE tokenizer not found. Train with --tokenizer bpe first.")
                sys.exit(1)
            tokenizer = load_tokenizer('model/spm')
            vocab_size = tokenizer.vocab_size
            pad_idx = tokenizer.pad_id
            print(f"Loaded BPE tokenizer (vocab: {vocab_size})")
        else:
            if not os.path.exists(src_vocab_path):
                print("Error: Vocabulary not found. Train first.")
                sys.exit(1)
            src_vocab = torch.load(src_vocab_path, weights_only=False)
            tgt_vocab = torch.load(tgt_vocab_path, weights_only=False)
            vocab_size = len(src_vocab)
            pad_idx = src_vocab['<pad>']
            print(f"Loaded vocab - src: {len(src_vocab)}, tgt: {len(tgt_vocab)}")

        print(f"Encoder: {args.encoder_type}, Tokenizer: {args.tokenizer}")
        encoder = create_encoder(EncoderConfig(
            encoder_type=args.encoder_type,
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_SIZE,
            padding_idx=pad_idx
        )).to(device)

        decoder_vocab_size = vocab_size if use_bpe else len(tgt_vocab)
        decoder_pad_idx = pad_idx if use_bpe else tgt_vocab['<pad>']
        decoder = Decoder(HIDDEN_SIZE, decoder_vocab_size, EMBEDDING_DIM, decoder_pad_idx).to(device)

        checkpoint = torch.load('model/trained_model.pth', map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        print(f"Loaded model (BLEU: {checkpoint.get('bleu', 'N/A'):.2f})")

        hf_dataset = load_multi30k()

        if use_bpe:
            test_dataset = Multi30kSubwordDataset(hf_dataset['test'], max_seq_length=MAX_SEQ_LENGTH)
            collate_fn = partial(collate_subword_batch, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)
        else:
            test_dataset = Multi30kDataset(hf_dataset['test'], max_seq_length=MAX_SEQ_LENGTH)
            collate_fn = partial(collate_batch, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                                src_lang=src_lang, tgt_lang=tgt_lang)

        print(f"Test pairs: {len(test_dataset)}")

        test_loader = create_optimized_dataloader(
            test_dataset, batch_size=64, shuffle=False,
            collate_fn=collate_fn, drop_last=False
        )

        vocab_for_decode = tokenizer if use_bpe else tgt_vocab
        bleu = validate(test_loader, encoder, decoder, vocab_for_decode, device)
        print(f"\nTest BLEU: {bleu:.2f}")

    elif args.command == 'train-orpo':
        # Load vocabularies from previous training
        if not os.path.exists(src_vocab_path):
            print("Error: Vocabulary not found. Train base model first.")
            sys.exit(1)

        src_vocab = torch.load(src_vocab_path, weights_only=False)
        tgt_vocab = torch.load(tgt_vocab_path, weights_only=False)
        print(f"Loaded vocab - src: {len(src_vocab)}, tgt: {len(tgt_vocab)}")

        # Load dataset
        hf_dataset = load_multi30k()
        full_dataset = Multi30kDataset(hf_dataset['train'], max_seq_length=MAX_SEQ_LENGTH)
        print(f"Training pairs: {len(full_dataset)}")

        train_size = int(0.95 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        batch_size = args.batch_size
        collate_fn = partial(collate_batch, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                            src_lang=src_lang, tgt_lang=tgt_lang)
        train_loader = create_optimized_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=True
        )
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, drop_last=False
        )

        # Initialize models
        print(f"\nEncoder: {args.encoder_type}")
        encoder = create_encoder(EncoderConfig(
            encoder_type=args.encoder_type,
            vocab_size=len(src_vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_SIZE,
            padding_idx=src_vocab['<pad>']
        )).to(device)

        decoder = Decoder(HIDDEN_SIZE, len(tgt_vocab), EMBEDDING_DIM, tgt_vocab['<pad>']).to(device)

        # Load pretrained checkpoint
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            print(f"Loaded checkpoint: {args.checkpoint} (BLEU: {checkpoint.get('bleu', 'N/A'):.2f})")
        else:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Starting from scratch.")

        enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
        dec_params = sum(p.numel() for p in decoder.parameters()) / 1e6
        print(f"Params - encoder: {enc_params:.2f}M, decoder: {dec_params:.2f}M")

        optimizer = torch.optim.Adam(
            itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr
        )
        lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        print(f"ORPO beta: {args.beta}")

        train_orpo(
            encoder=encoder,
            decoder=decoder,
            train_data_loader=train_loader,
            val_data_loader=val_loader,
            optimizer=optimizer,
            target_vocab=tgt_vocab,
            lr_scheduler=lr_scheduler,
            num_epochs=args.epochs,
            device=device,
            beta=args.beta
        )

    elif args.command == 'train-enhanced':
        # Enhanced training with all PhD-level improvements
        batch_size = args.batch_size
        num_epochs = args.epochs
        learning_rate = args.lr
        use_bpe = args.tokenizer == 'bpe'
        use_deep_decoder = args.decoder_type == 'deep'

        # Load dataset
        hf_dataset = load_multi30k()

        if use_bpe:
            full_dataset = Multi30kSubwordDataset(hf_dataset['train'], max_seq_length=MAX_SEQ_LENGTH)
            print(f"Training pairs: {len(full_dataset)}")

            train_size = int(0.95 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Train or load BPE tokenizer
            tokenizer_path = 'model/spm.model'
            os.makedirs('model', exist_ok=True)

            if os.path.exists(tokenizer_path):
                print(f"Loading existing BPE tokenizer from {tokenizer_path}")
                tokenizer = load_tokenizer('model/spm')
            else:
                print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
                tokenizer = create_joint_tokenizer(
                    full_dataset.src_texts,
                    full_dataset.tgt_texts,
                    vocab_size=args.vocab_size,
                    model_prefix='model/spm',
                    verbose=True
                )

            vocab_size = tokenizer.vocab_size
            pad_idx = tokenizer.pad_id
            print(f"BPE vocab size: {vocab_size}")

            collate_fn = partial(collate_subword_batch, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH)
            target_vocab = tokenizer
        else:
            full_dataset = Multi30kDataset(hf_dataset['train'], max_seq_length=MAX_SEQ_LENGTH)
            print(f"Training pairs: {len(full_dataset)}")

            train_size = int(0.95 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            src_vocab = build_vocabulary(full_dataset.src_sequences, src_lang)
            tgt_vocab = build_vocabulary(full_dataset.tgt_sequences, tgt_lang)
            print(f"Vocab sizes - src: {len(src_vocab)}, tgt: {len(tgt_vocab)}")

            vocab_size = len(src_vocab)
            pad_idx = src_vocab['<pad>']

            collate_fn = partial(collate_batch, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                                src_lang=src_lang, tgt_lang=tgt_lang)
            target_vocab = tgt_vocab

            os.makedirs('model', exist_ok=True)
            torch.save(src_vocab, src_vocab_path)
            torch.save(tgt_vocab, tgt_vocab_path)

        train_loader = create_optimized_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=True
        )
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, drop_last=False
        )

        # Initialize models
        print(f"\nEncoder: {args.encoder_type}, Decoder: {args.decoder_type}, Tokenizer: {args.tokenizer}")
        encoder = create_encoder(EncoderConfig(
            encoder_type=args.encoder_type,
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_SIZE,
            padding_idx=pad_idx
        )).to(device)

        decoder_vocab_size = vocab_size if use_bpe else len(tgt_vocab)
        decoder_pad_idx = pad_idx if use_bpe else tgt_vocab['<pad>']

        if use_deep_decoder:
            decoder = DeepDecoder(
                hidden_dim=HIDDEN_SIZE,
                vocab_size=decoder_vocab_size,
                embedding_dim=EMBEDDING_DIM,
                padding_idx=decoder_pad_idx,
                num_layers=3,
                num_heads=8,
                ffn_dim=2048,
                use_coverage=not args.no_coverage
            ).to(device)
        else:
            decoder = Decoder(HIDDEN_SIZE, decoder_vocab_size, EMBEDDING_DIM, decoder_pad_idx).to(device)

        enc_params = sum(p.numel() for p in encoder.parameters()) / 1e6
        dec_params = sum(p.numel() for p in decoder.parameters()) / 1e6
        print(f"Params - encoder: {enc_params:.2f}M, decoder: {dec_params:.2f}M")

        # Use AdamW for Transformer
        optimizer = torch.optim.AdamW(
            itertools.chain(encoder.parameters(), decoder.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Run enhanced training
        train_enhanced(
            encoder=encoder,
            decoder=decoder,
            train_data_loader=train_loader,
            val_data_loader=val_loader,
            optimizer=optimizer,
            target_vocab=target_vocab,
            num_epochs=num_epochs,
            device=device,
            # Training enhancements
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            use_rdrop=not args.no_rdrop,
            rdrop_alpha=args.rdrop_alpha,
            use_coverage=not args.no_coverage and use_deep_decoder,
            coverage_lambda=args.coverage_lambda,
            # Optimization
            peak_lr=learning_rate,
            # Validation
            use_beam_search=not args.no_beam_search,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty
        )


if __name__ == '__main__':
    main()
