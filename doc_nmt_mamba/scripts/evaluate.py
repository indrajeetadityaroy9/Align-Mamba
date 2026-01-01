#!/usr/bin/env python3
"""
Evaluation script for Document-Level NMT with Hybrid Mamba-Attention.

Usage:
    python scripts/evaluate.py checkpoint=outputs/checkpoint-10000
    python scripts/evaluate.py checkpoint=outputs/best --eval_mode=full
    python scripts/evaluate.py --eval_mode=length_analysis

Evaluation modes:
- standard: BLEU, chrF, TER, COMET
- contrapro: Contrastive pronoun evaluation
- entity: Named entity recall analysis
- length: Length sensitivity analysis
- full: All of the above
"""

import os
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

from models import ModelConfig, HybridMambaEncoderDecoder
from data import NMTTokenizer, IWSLT14Dataset, create_collator
from evaluation import (
    EvaluationSuite,
    evaluate_pronoun_accuracy,
    create_synthetic_contrapro_examples,
    analyze_entity_recall,
    analyze_length_sensitivity,
    LengthSensitivityAnalyzer,
    ExtrapolationTester,
)


def load_model(
    checkpoint_path: str,
    config: DictConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> HybridMambaEncoderDecoder:
    """Load model from checkpoint."""
    model_cfg = ModelConfig(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        encoder_layers=config.model.encoder_layers,
        decoder_layers=config.model.decoder_layers,
        d_state=config.model.d_state,
        n_heads=config.model.n_heads,
        attention_ratio=config.model.attention_ratio,
        cross_attn_every=config.model.cross_attn_every,
        dropout=0.0,  # No dropout for evaluation
        max_seq_len=config.model.max_seq_len,
    )

    model = HybridMambaEncoderDecoder(
        config=model_cfg,
        device=device,
        dtype=dtype,
    )

    # Load checkpoint
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        model_file = checkpoint / "model.pt"
    else:
        model_file = checkpoint

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")

    return model


@torch.no_grad()
def generate_translations(
    model: HybridMambaEncoderDecoder,
    tokenizer: NMTTokenizer,
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
        encoded = tokenizer.encode_source(batch_sources)
        src_ids = encoded["input_ids"].to(device)

        # Generate
        generated = model.generate(src_ids, max_length=max_length)

        # Decode
        batch_translations = tokenizer.batch_decode(generated)
        translations.extend(batch_translations)

    return translations


def evaluate_standard(
    model: HybridMambaEncoderDecoder,
    tokenizer: NMTTokenizer,
    test_dataset,
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict:
    """Run standard MT evaluation (BLEU, COMET, etc.)."""
    print("\n" + "=" * 60)
    print("Standard MT Evaluation")
    print("=" * 60)

    # Extract sources and references
    sources = []
    references = []
    for i in range(len(test_dataset)):
        sample = test_dataset.samples[i]
        sources.append(sample[0])
        references.append(sample[1])

    # Generate translations
    translations = generate_translations(
        model, tokenizer, sources, batch_size, device=device
    )

    # Evaluate
    eval_suite = EvaluationSuite(use_comet=True)
    result = eval_suite.evaluate(sources, translations, references)

    print(f"\nResults:")
    print(f"  BLEU:  {result.bleu:.2f}")
    print(f"  chrF:  {result.chrf:.2f}")
    print(f"  TER:   {result.ter:.2f}")
    print(f"  COMET: {result.comet:.4f}")

    return {
        "bleu": result.bleu,
        "chrf": result.chrf,
        "ter": result.ter,
        "comet": result.comet,
        "n_samples": len(sources),
    }


def evaluate_contrapro(
    model: HybridMambaEncoderDecoder,
    tokenizer: NMTTokenizer,
    contrapro_path: Optional[str] = None,
    device: str = "cuda",
) -> Dict:
    """Run contrastive pronoun evaluation."""
    print("\n" + "=" * 60)
    print("Contrastive Pronoun Evaluation (ContraPro)")
    print("=" * 60)

    # Load or create examples
    if contrapro_path and Path(contrapro_path).exists():
        from evaluation import ContraProDataset
        dataset = ContraProDataset(contrapro_path)
        examples = list(dataset)
    else:
        print("Using synthetic examples (real ContraPro data not found)")
        examples = create_synthetic_contrapro_examples(100)

    # Evaluate
    result = evaluate_pronoun_accuracy(model, tokenizer, examples, device)

    print(f"\nResults:")
    print(f"  Overall Accuracy: {result.accuracy:.2%}")
    print(f"  Correct: {result.correct}/{result.total_examples}")

    if result.by_pronoun_type:
        print("\n  By Pronoun Type:")
        for ptype, acc in result.by_pronoun_type.items():
            print(f"    {ptype}: {acc:.2%}")

    if result.by_distance:
        print("\n  By Antecedent Distance:")
        for dist, acc in sorted(result.by_distance.items()):
            print(f"    Distance {dist}: {acc:.2%}")

    return {
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total_examples,
        "by_pronoun_type": result.by_pronoun_type,
        "by_distance": result.by_distance,
    }


def evaluate_entity_recall(
    model: HybridMambaEncoderDecoder,
    tokenizer: NMTTokenizer,
    test_dataset,
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict:
    """Run named entity recall analysis."""
    print("\n" + "=" * 60)
    print("Named Entity Recall Analysis")
    print("=" * 60)

    # Get sources and generate translations
    sources = [test_dataset.samples[i][0] for i in range(min(500, len(test_dataset)))]
    translations = generate_translations(
        model, tokenizer, sources, batch_size, device=device
    )

    # Analyze
    result = analyze_entity_recall(sources, translations)

    print(f"\nResults:")
    print(f"  Overall Recall: {result.overall_recall:.2%}")
    print(f"  Total Source Entities: {result.total_source_entities}")
    print(f"  Found in Translation: {result.total_found_in_translation}")

    if result.by_entity_type:
        print("\n  By Entity Type:")
        for etype, metrics in result.by_entity_type.items():
            print(f"    {etype}: {metrics['recall']:.2%} ({metrics['total']} total)")

    if result.missing_entities:
        print(f"\n  Most Common Missing Entities: {result.missing_entities[:10]}")

    return {
        "recall": result.overall_recall,
        "total_entities": result.total_source_entities,
        "found": result.total_found_in_translation,
        "by_type": result.by_entity_type,
    }


def evaluate_length_sensitivity(
    model: HybridMambaEncoderDecoder,
    tokenizer: NMTTokenizer,
    test_dataset=None,
    device: str = "cuda",
) -> Dict:
    """Run length sensitivity analysis."""
    print("\n" + "=" * 60)
    print("Length Sensitivity Analysis")
    print("=" * 60)

    # Memory and speed scaling
    print("\nMeasuring memory and speed scaling...")
    analyzer = LengthSensitivityAnalyzer(model, tokenizer, device)

    test_lengths = [128, 256, 512, 1024, 2048]
    memory_scaling, speed_scaling = analyzer.analyze_scaling(test_lengths)

    print("\nMemory Scaling:")
    for length, memory in memory_scaling.items():
        print(f"  {length} tokens: {memory:.2f} GB")

    print("\nSpeed Scaling (tokens/sec):")
    for length, speed in speed_scaling.items():
        print(f"  {length} tokens: {speed:.1f}")

    # Extrapolation test
    print("\nTesting length extrapolation...")
    extrapolation_tester = ExtrapolationTester(
        model, tokenizer,
        training_max_length=2048,
        device=device,
    )
    extrapolation_results = extrapolation_tester.test_extrapolation()

    print("\nExtrapolation Results:")
    for length, result in extrapolation_results.items():
        status = "OK" if result["success"] else "FAILED"
        relative = result["relative_to_training"]
        print(f"  {length} tokens ({relative:.1f}x training): {status}")

    return {
        "memory_scaling": memory_scaling,
        "speed_scaling": speed_scaling,
        "extrapolation": extrapolation_results,
    }


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
    dtype = torch.bfloat16 if cfg.training.use_bf16 else torch.float32

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {cfg.checkpoint}")

    # Create tokenizer
    print("\nLoading tokenizer...")
    tokenizer = NMTTokenizer(
        src_lang=cfg.data.src_lang,
        tgt_lang=cfg.data.tgt_lang,
    )
    cfg.model.vocab_size = tokenizer.vocab_size

    # Load model
    print("\nLoading model...")
    model = load_model(cfg.checkpoint, cfg, device, dtype)

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = IWSLT14Dataset(
        split="test",
        tokenizer=tokenizer,
        max_src_length=cfg.data.max_src_length,
        max_tgt_length=cfg.data.max_tgt_length,
    )
    print(f"Test samples: {len(test_dataset)}")

    # Determine evaluation mode
    eval_mode = cfg.get("eval_mode", "standard")
    results = {}

    if eval_mode in ["standard", "full"]:
        results["standard"] = evaluate_standard(
            model, tokenizer, test_dataset,
            batch_size=cfg.training.batch_size,
            device=device,
        )

    if eval_mode in ["contrapro", "full"]:
        results["contrapro"] = evaluate_contrapro(
            model, tokenizer,
            contrapro_path=cfg.get("contrapro_path"),
            device=device,
        )

    if eval_mode in ["entity", "full"]:
        results["entity_recall"] = evaluate_entity_recall(
            model, tokenizer, test_dataset,
            batch_size=cfg.training.batch_size,
            device=device,
        )

    if eval_mode in ["length", "full"]:
        results["length_sensitivity"] = evaluate_length_sensitivity(
            model, tokenizer, test_dataset,
            device=device,
        )

    # Save results
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "evaluation_results.json"

    # Convert numpy/tensor values for JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        else:
            return obj

    with open(results_file, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"Results saved to {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
