# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural Machine Translation (NMT) pipeline for German-to-English translation using an encoder-decoder RNN architecture with Bahdanau attention. Uses the Multi30k dataset (~29k training pairs) - a standard NMT benchmark.

Supports two encoder architectures for research comparison:
- **BiGRU** (baseline): Bidirectional GRU encoder
- **Mamba-2 SSM**: State space model with bidirectional blocks

## Commands

```bash
# Train with BiGRU encoder (baseline)
python3 NMT.py train --encoder-type bigru --epochs 10

# Train with Mamba-2 encoder
python3 NMT.py train --encoder-type mamba2 --epochs 10

# Test trained model
python3 NMT.py test --encoder-type bigru
```

## Architecture

### Project Structure
```
.
├── NMT.py              # CLI entry point
├── models/
│   ├── encoder.py      # BiGRU and Mamba-2 encoders
│   └── decoder.py      # GRU decoder with Bahdanau attention
├── data/
│   ├── vocab.py        # Vocabulary class
│   ├── tokenizer.py    # German/English tokenization
│   └── dataset.py      # Multi30kDataset and collation
└── training/
    ├── trainer.py      # Training loop
    └── evaluator.py    # BLEU evaluation
```

### Model Components
- **Encoder**: BiGRU or Mamba-2 SSM with bidirectional context
- **Decoder**: GRU with Bahdanau attention mechanism
- **Dataset**: Multi30k German-English (~29k training pairs)

### Key Hyperparameters
```python
MAX_SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 512
HIDDEN_SIZE = 1024
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
```

### Training Features
- BFloat16 mixed precision training
- Gradient accumulation (4 steps)
- StepLR scheduler
- Model checkpointing based on BLEU score

### Saved Artifacts
- `model/trained_model.pth`: Best encoder/decoder weights
- `model/src_vocab.pth`: Source (German) vocabulary
- `model/tgt_vocab.pth`: Target (English) vocabulary

### Dependencies
- torch, mamba-ssm, causal-conv1d
- transformers, datasets
- nltk (English tokenization)
- tqdm
