# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document-Level Neural Machine Translation using Hybrid Mamba-2/Attention architecture. Combines causal Mamba blocks with sparse cross-attention layers (1:8 ratio with HYBRID blocks at layers [0, 8, 16]) for O(L) complexity on H100 hardware. Targets coherent document translation with pronoun/entity consistency.

## Environment Setup

```bash
# Activate virtual environment (always do this first)
source venv/bin/activate

# Install base dependencies (CPU-compatible)
pip install -e ".[dev]"

# On H100/CUDA machine, also install GPU packages:
pip install -e ".[cuda]"  # mamba-ssm, causal-conv1d
pip install -e ".[flash]" # flash-attn

# All optional dependencies
pip install -e ".[all]"
```

Virtual environment: `venv/` (Python 3.10.11)
Requires CUDA 12.3+ for Mamba-2 kernels. FlashAttention-2 has PyTorch SDPA fallback.

## Commands

### Testing
```bash
# All tests (from project root, with venv activated)
python -m pytest doc_nmt_mamba/tests/ -v

# Individual test files
python -m pytest doc_nmt_mamba/tests/test_models.py -v
python -m pytest doc_nmt_mamba/tests/test_synthetic.py -v
python -m pytest doc_nmt_mamba/tests/test_verification_checklist.py -v
python -m pytest doc_nmt_mamba/tests/test_data_pipeline.py -v
```

### Training
```bash
# Single GPU (defaults to medium model - 200M params)
python doc_nmt_mamba/scripts/train.py

# With specific model size
python doc_nmt_mamba/scripts/train.py model=medium    # 200M params (primary)
python doc_nmt_mamba/scripts/train.py model=small     # 25M params (debugging)
python doc_nmt_mamba/scripts/train.py model=base      # 77M params

# Multi-GPU DDP
torchrun --nproc_per_node=2 doc_nmt_mamba/scripts/train.py

# Debug mode
python doc_nmt_mamba/scripts/train.py model=small training=debug
```

### Evaluation
```bash
# Full publication evaluation
python doc_nmt_mamba/scripts/evaluation/run_full_evaluation.py --checkpoint path/model.pt

# Quick evaluation (skip COMET)
python doc_nmt_mamba/scripts/evaluation/run_full_evaluation.py --checkpoint path/model.pt --quick
```

### Utilities
```bash
# Build 32K BPE tokenizer
python doc_nmt_mamba/scripts/build_tokenizer.py

# Benchmark hardware performance
python doc_nmt_mamba/scripts/benchmark_hardware.py
```

## Architecture

### Hybrid Design (1:8 Ratio with HYBRID Blocks)

**Encoder**: BiMamba (forward+backward scan) + bidirectional attention at layers N/2 and N-1

**Decoder** (24 layers):
- Layer 0: **HYBRID BLOCK** (Mamba + Cross-Attn) - Contextualized Preamble
- Layers 1-7: Mamba only (causal)
- Layer 8: **HYBRID BLOCK** (Mamba + Cross-Attn) - Refresh 1
- Layers 9-15: Mamba only (causal)
- Layer 16: **HYBRID BLOCK** (Mamba + Cross-Attn) - Refresh 2
- Layers 17-23: Mamba only (causal)

Each HYBRID block:
```
x = x + Mamba(RMSNorm(x))           # Position-aware queries
x = x + CrossAttn(RMSNorm(x), enc)  # Source-aligned output
```

### Key Modules
```
doc_nmt_mamba/
├── models/
│   ├── mamba2/           # Mamba-2 SSM blocks (conditional CUDA imports)
│   │   ├── norms.py      # RMSNorm (always available)
│   │   ├── bimamba.py    # BiMambaBlock, segment_aware_flip
│   │   └── mamba2_wrapper.py  # Mamba2BlockWrapper
│   ├── attention/        # FlashAttention-2 with SDPA fallback
│   │   ├── rope.py       # Rotary Position Embedding
│   │   ├── flash_cross_attention.py
│   │   ├── causal_self_attention.py
│   │   └── bidirectional_attention.py
│   ├── hybrid/           # Layer placement logic
│   │   ├── layer_builder.py  # build_encoder_layers, build_decoder_layers
│   │   ├── encoder.py    # HybridBiMambaEncoder
│   │   └── decoder.py    # HybridMambaDecoder
│   └── encoder_decoder.py  # HybridMambaEncoderDecoder, ModelConfig
├── data/
│   ├── synthetic.py      # MQARDataset for state capacity testing
│   ├── augmentation.py   # ConcatenationAugmenter (CAT-N)
│   ├── collation.py      # PackedSequenceCollator
│   ├── tokenization.py   # NMTTokenizer (32K BPE)
│   └── document_dataset.py
├── evaluation/
│   ├── metrics.py        # BLEUScorer, CHRFScorer, COMETScorer
│   ├── alignment.py      # SubwordToWordMapper, AlignmentEvaluator
│   ├── contrapro.py      # Contrastive pronoun evaluation
│   └── entity_recall.py  # Entity consistency analysis
├── training/
│   ├── trainer.py        # Trainer, TrainerConfig
│   ├── hardware.py       # H100 optimization, GPU detection
│   ├── distributed.py    # DDP/FSDP setup
│   └── schedulers.py     # Learning rate schedulers
├── tests/
│   ├── test_verification_checklist.py  # Critical mechanism tests
│   ├── test_models.py
│   ├── test_synthetic.py
│   └── test_data_pipeline.py
└── configs/              # Hydra configuration
    ├── config.yaml       # Main entry point
    ├── model/            # small/base/medium/large
    ├── training/         # default/debug/fast
    └── data/             # dataset configs
```

## Critical Technical Decisions

1. **HYBRID Blocks at Layer 0**: First decoder layer MUST be HYBRID (Mamba + Cross-Attn) to fix "Blind Start" problem. Pure cross-attention at layer 0 lacks positional context.

2. **Conditional CUDA Imports**: All `mamba_ssm` imports are wrapped in try/except. Code is importable on CPU for development/testing.

3. **BiMamba for Encoder**: Forward+backward scans with `segment_aware_flip` that respects document boundaries (cu_seqlens).

4. **Use mamba-ssm library**: Do NOT reimplement SSD algorithm - CUDA kernels are 10-50x faster than PyTorch.

5. **Custom 32K BPE Tokenizer**: Not mBART's 250K vocab (wastes 95% of embedding table).

6. **RMSNorm in Mamba blocks**: Required for training stability at 200M+ parameters.

7. **CAT-N Augmentation**: Critical for length generalization. 50% single sentences, 50% CAT-5 concatenated with `<doc>` separator.

8. **SubwordToWordMapper**: Required for AER computation - maps BPE tokens back to word indices.

## Verification Checklist (Pre-Training)

Run before H100 training:
```bash
python -m pytest doc_nmt_mamba/tests/test_verification_checklist.py -v
```

Tests verify:
- segment_aware_flip respects document boundaries
- HYBRID blocks at [0, 8, 16]
- MQAR dataset has no leakage
- SubwordToWordMapper for AER
- CAT-N concatenation with separators

## Optional Dependencies

```bash
pip install -e ".[cuda]"   # mamba-ssm, causal-conv1d (CUDA only)
pip install -e ".[flash]"  # flash-attn (H100 recommended)
pip install -e ".[comet]"  # COMET neural evaluation
pip install -e ".[nlp]"    # spaCy for entity analysis
pip install -e ".[viz]"    # matplotlib/seaborn plotting
pip install -e ".[dev]"    # pytest, black, mypy
pip install -e ".[all]"    # Everything (GPU environment)
```
