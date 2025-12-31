# Neural Machine Translation: German to English

Encoder-decoder RNN architecture with Bahdanau attention for German-to-English translation. Supports BiGRU (baseline) and Mamba-2 SSM encoders for architecture comparison research.

## Dataset

**Multi30k** (~29k training pairs) - Standard NMT benchmark for German-English translation.

## Features

- **Dual Encoder Support**: BiGRU baseline and Mamba-2 SSM with bidirectional blocks
- **Bahdanau Attention**: Context-aware decoding with learned alignment
- **BFloat16 Training**: Mixed precision for H100 optimization
- **Gradient Accumulation**: Effective batch size scaling
- **BLEU Evaluation**: Corpus-level scoring with checkpointing

## Usage

```bash
# Train with BiGRU encoder
python3 NMT.py train --encoder-type bigru --epochs 10

# Train with Mamba-2 encoder
python3 NMT.py train --encoder-type mamba2 --epochs 10

# Test trained model
python3 NMT.py test --encoder-type bigru
```

## Architecture

| Component | Description |
|-----------|-------------|
| Encoder | BiGRU or Mamba-2 SSM (bidirectional) |
| Decoder | GRU with Bahdanau attention |
| Embedding | 512 dimensions |
| Hidden | 1024 dimensions |

## Dependencies

```
torch>=2.0
mamba-ssm>=2.2.0
causal-conv1d>=1.4.0
transformers
datasets
nltk
tqdm
```
