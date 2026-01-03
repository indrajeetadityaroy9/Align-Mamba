#!/usr/bin/env python3
"""
Unified Evaluation Script for Document-Level NMT.

This is a thin wrapper that delegates to the evaluation library.

Usage:
    # Full evaluation
    python scripts/evaluate.py checkpoint=outputs/checkpoint-10000

    # Specific modes
    python scripts/evaluate.py checkpoint=outputs/best eval_mode=quality
    python scripts/evaluate.py checkpoint=outputs/best eval_mode=contrapro
    python scripts/evaluate.py checkpoint=outputs/best eval_mode=efficiency
    python scripts/evaluate.py checkpoint=outputs/best eval_mode=full

    # Quick evaluation (skip COMET, fewer samples)
    python scripts/evaluate.py checkpoint=outputs/best quick=true

Evaluation modes:
- quality: BLEU, chrF, TER, COMET
- contrapro: Contrastive pronoun evaluation
- efficiency: Throughput and memory benchmarks
- full: All of the above
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from models import ModelConfig, HybridMambaEncoderDecoder, load_model_from_checkpoint
from data import IWSLT14Dataset, create_tokenizer
from evaluation import (
    # Runner
    EvaluationRunner,
    RunnerConfig,
    # Metrics
    EvaluationSuite,
    # Analysis
    ContrastivePronounEvaluator,
    ContraProDataset,
    analyze_entity_recall,
    LengthSensitivityAnalyzer,
    ExtrapolationTester,
)


@torch.no_grad()
def generate_translations(
    model: HybridMambaEncoderDecoder,
    tokenizer,
    sources: List[str],
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "cuda",
) -> List[str]:
    """Generate translations for a list of source sentences."""
    translations = []

    for i in tqdm(range(0, len(sources), batch_size), desc="Translating"):
        batch_sources = sources[i : i + batch_size]

        # Encode sources
        encoded = tokenizer(batch_sources, return_tensors="pt", padding=True)
        src_ids = encoded["input_ids"].to(device)

        # Generate
        generated = model.generate(src_ids, max_length=max_length)

        # Decode
        batch_translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(batch_translations)

    return translations


def load_test_data(dataset_name: str, split: str = "test", **kwargs):
    """Load test data sources and references."""
    if dataset_name.lower() == "iwslt14":
        # Load IWSLT14 test data
        data_path = Path("data/iwslt14/test.de-en")
        sources = []
        references = []

        src_file = data_path.parent / f"test.de"
        ref_file = data_path.parent / f"test.en"

        if src_file.exists() and ref_file.exists():
            with open(src_file) as f:
                sources = [line.strip() for line in f]
            with open(ref_file) as f:
                references = [line.strip() for line in f]
        else:
            print(f"Test files not found at {data_path.parent}")
            print("Using dummy data for demonstration")
            sources = ["Hallo Welt."] * 10
            references = ["Hello World."] * 10

        return sources, references

    raise ValueError(f"Unknown dataset: {dataset_name}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main evaluation function."""
    print("=" * 60)
    print("Document-Level NMT Evaluation")
    print("=" * 60)

    # Check for checkpoint
    if not cfg.get("checkpoint"):
        print("No checkpoint specified. Use: python evaluate.py checkpoint=<path>")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if cfg.training.get("use_bf16", True) else torch.float32

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {cfg.checkpoint}")

    # Determine mode
    eval_mode = cfg.get("eval_mode", "quality")
    quick = cfg.get("quick", False)

    # Create runner config
    runner_config = RunnerConfig(
        output_dir=cfg.get("output_dir", "outputs/evaluation"),
        skip_comet=cfg.get("skip_comet", False),
        quick=quick,
    )

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = create_tokenizer(
        tokenizer_type="custom",
        tokenizer_path="data/tokenizer/tokenizer.json",
    )

    # Load model
    print("\nLoading model...")
    try:
        model, model_config = load_model_from_checkpoint(
            cfg.checkpoint,
            device=device,
            dtype=dtype,
        )
        model.eval()
        print(f"Loaded model: {model.num_parameters() / 1e6:.1f}M parameters")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure checkpoint path is correct and contains model.pt or model.v2.pt")
        return

    model_name = Path(cfg.checkpoint).stem

    # Create runner
    runner = EvaluationRunner(runner_config, device=device)

    # Determine what to run
    run_quality = eval_mode in ["quality", "full"]
    run_efficiency = eval_mode in ["efficiency", "full"]
    run_contrapro = eval_mode in ["contrapro", "full"]

    # Load test data for quality evaluation
    sources, hypotheses, references = None, None, None
    if run_quality:
        print("\nLoading test data...")
        sources, references = load_test_data("iwslt14", "test")
        print(f"Test samples: {len(sources)}")

        # Generate translations
        print("\nGenerating translations...")
        hypotheses = generate_translations(
            model, tokenizer, sources,
            batch_size=cfg.training.get("batch_size", 16),
            max_length=cfg.data.get("max_tgt_length", 256),
            device=device,
        )

    # Run evaluation
    result = runner.run_full_evaluation(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        hypotheses=hypotheses,
        references=references,
        model_name=model_name,
        run_quality=run_quality,
        run_efficiency=run_efficiency,
        run_contrapro=run_contrapro,
    )

    print(f"\nResults saved to: {runner_config.output_dir}")


if __name__ == "__main__":
    main()
