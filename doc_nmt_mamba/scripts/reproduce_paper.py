#!/usr/bin/env python3
"""
Reproduce Paper Results - Bash Script Generator

Generates shell scripts for running all paper experiments.
This is NOT a subprocess orchestrator - it generates scripts that
reviewers can read, audit, and execute manually.

Usage:
    # Preview commands (dry-run)
    python scripts/reproduce_paper.py --dry-run

    # Generate runnable script
    python scripts/reproduce_paper.py --generate-script

    # Generate only main results (Table 1)
    python scripts/reproduce_paper.py --main-results --generate-script

    # Generate only ablations (Table 2)
    python scripts/reproduce_paper.py --ablations --generate-script

    # Then run the generated script:
    bash run_experiments.sh

    # Or parallelize manually:
    sed -n '1,4p' run_experiments.sh | parallel
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Experiment:
    """Experiment configuration."""

    name: str
    config: str
    paper_ref: str
    category: str  # main, ablation, mqar, scaling, eval
    gpu_memory_gb: int = 40  # Estimated GPU memory needed
    description: str = ""


# All paper experiments
EXPERIMENTS = [
    # Main Results (Table 1)
    Experiment(
        name="main_iwslt_baseline",
        config="experiment=main_iwslt_baseline",
        paper_ref="Table 1, Row 1",
        category="main",
        gpu_memory_gb=40,
        description="Transformer baseline on IWSLT14",
    ),
    Experiment(
        name="main_opus_baseline",
        config="experiment=main_opus_baseline",
        paper_ref="Table 1, Row 2",
        category="main",
        gpu_memory_gb=40,
        description="Transformer baseline on OPUS Books",
    ),
    Experiment(
        name="main_iwslt_hybrid",
        config="experiment=main_iwslt_hybrid",
        paper_ref="Table 1, Row 3",
        category="main",
        gpu_memory_gb=40,
        description="Align-Mamba on IWSLT14",
    ),
    Experiment(
        name="main_opus_hybrid",
        config="experiment=main_opus_hybrid",
        paper_ref="Table 1, Row 4",
        category="main",
        gpu_memory_gb=40,
        description="Align-Mamba on OPUS Books",
    ),
    # Ablations (Table 2)
    Experiment(
        name="ablation_no_hybrid_layer0",
        config="experiment=ablation_no_hybrid_layer0",
        paper_ref="Table 2 - Blind Start",
        category="ablation",
        gpu_memory_gb=40,
        description="HYBRID at [8,16] only (no layer 0)",
    ),
    Experiment(
        name="ablation_all_attention",
        config="experiment=ablation_all_attention",
        paper_ref="Table 2 - Pure Attention",
        category="ablation",
        gpu_memory_gb=40,
        description="100% attention (Transformer)",
    ),
    Experiment(
        name="ablation_no_cross_attn",
        config="experiment=ablation_no_cross_attn",
        paper_ref="Table 2 - Pure Mamba",
        category="ablation",
        gpu_memory_gb=40,
        description="No HYBRID blocks (pure Mamba)",
    ),
    Experiment(
        name="ablation_hybrid_ratio_4",
        config="experiment=ablation_hybrid_ratio_4",
        paper_ref="Table 2 - Ratio 1:4",
        category="ablation",
        gpu_memory_gb=40,
        description="HYBRID every 4 layers",
    ),
    Experiment(
        name="ablation_hybrid_ratio_16",
        config="experiment=ablation_hybrid_ratio_16",
        paper_ref="Table 2 - Ratio 1:16",
        category="ablation",
        gpu_memory_gb=40,
        description="HYBRID every 16 layers",
    ),
    Experiment(
        name="ablation_layer0_only",
        config="experiment=ablation_layer0_only",
        paper_ref="Table 2 - Layer 0 Only",
        category="ablation",
        gpu_memory_gb=40,
        description="HYBRID at layer 0 only",
    ),
    # MQAR (Figure 1)
    Experiment(
        name="mqar_state_sweep",
        config="experiment=mqar_state_sweep",
        paper_ref="Figure 1",
        category="mqar",
        gpu_memory_gb=20,
        description="State capacity sweep with MQAR",
    ),
    # Scaling (Figure 3)
    Experiment(
        name="scaling_small",
        config="experiment=scaling_small",
        paper_ref="Figure 3 - 25M",
        category="scaling",
        gpu_memory_gb=20,
        description="25M parameter model",
    ),
    Experiment(
        name="scaling_base",
        config="experiment=scaling_base",
        paper_ref="Figure 3 - 77M",
        category="scaling",
        gpu_memory_gb=30,
        description="77M parameter model",
    ),
    Experiment(
        name="scaling_medium",
        config="experiment=scaling_medium",
        paper_ref="Figure 3 - 200M",
        category="scaling",
        gpu_memory_gb=40,
        description="200M parameter model (primary)",
    ),
    Experiment(
        name="scaling_large",
        config="experiment=scaling_large",
        paper_ref="Figure 3 - 400M",
        category="scaling",
        gpu_memory_gb=60,
        description="400M parameter model",
    ),
    # ContraPro Evaluation (Table 3 / Figure 4)
    Experiment(
        name="contrapro_eval",
        config="experiment=contrapro_eval",
        paper_ref="Table 3 / Figure 4",
        category="eval",
        gpu_memory_gb=40,
        description="ContraPro pronoun evaluation",
    ),
]


def get_train_command(exp: Experiment, multi_gpu: bool = False) -> str:
    """Generate training command for an experiment."""
    if multi_gpu:
        # DDP with 2 GPUs
        cmd = f"torchrun --nproc_per_node=2 doc_nmt_mamba/scripts/train.py {exp.config}"
    else:
        # Single GPU
        cmd = f"python doc_nmt_mamba/scripts/train.py {exp.config}"

    return cmd


def get_eval_command(exp: Experiment, checkpoint: str = "auto") -> str:
    """Generate evaluation command for an experiment."""
    if checkpoint == "auto":
        checkpoint = f"outputs/{exp.name}/best_model.pt"

    cmd = f"python doc_nmt_mamba/scripts/evaluate.py --checkpoint {checkpoint}"
    return cmd


def filter_experiments(
    experiments: List[Experiment],
    categories: Optional[List[str]] = None,
    names: Optional[List[str]] = None,
) -> List[Experiment]:
    """Filter experiments by category or name."""
    if categories:
        experiments = [e for e in experiments if e.category in categories]
    if names:
        experiments = [e for e in experiments if e.name in names]
    return experiments


def generate_script(
    experiments: List[Experiment],
    multi_gpu: bool = False,
    output_file: str = "run_experiments.sh",
) -> str:
    """Generate bash script for running experiments."""
    lines = [
        "#!/bin/bash",
        "#",
        f"# Align-Mamba Paper Reproduction Script",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Experiments: {len(experiments)}",
        "#",
        "# Usage:",
        "#   bash run_experiments.sh",
        "#",
        "# For parallel execution on multiple GPUs:",
        "#   sed -n '1,4p' run_experiments.sh | CUDA_VISIBLE_DEVICES=0 bash",
        "#   sed -n '5,8p' run_experiments.sh | CUDA_VISIBLE_DEVICES=1 bash",
        "#",
        "",
        "set -e  # Exit on error",
        "",
        "# Activate virtual environment",
        "source venv/bin/activate",
        "",
    ]

    for i, exp in enumerate(experiments, 1):
        lines.extend(
            [
                f"# [{i}/{len(experiments)}] {exp.paper_ref}: {exp.description}",
                f"echo '>>> Running: {exp.name} ({exp.paper_ref})'",
                get_train_command(exp, multi_gpu),
                "",
            ]
        )

    lines.extend(
        [
            "echo '>>> All experiments complete!'",
            "",
        ]
    )

    script = "\n".join(lines)
    return script


def print_summary(experiments: List[Experiment]) -> None:
    """Print summary of experiments to run."""
    print("\n" + "=" * 70)
    print("Align-Mamba Paper Reproduction")
    print("=" * 70)

    # Group by category
    categories = {}
    for exp in experiments:
        if exp.category not in categories:
            categories[exp.category] = []
        categories[exp.category].append(exp)

    category_names = {
        "main": "Main Results (Table 1)",
        "ablation": "Ablations (Table 2)",
        "mqar": "MQAR State Capacity (Figure 1)",
        "scaling": "Scaling Experiments (Figure 3)",
        "eval": "Evaluation (Table 3 / Figure 4)",
    }

    for cat, exps in categories.items():
        print(f"\n{category_names.get(cat, cat)}:")
        for exp in exps:
            print(f"  - {exp.name}: {exp.description}")

    total_gpu_hours = sum(e.gpu_memory_gb for e in experiments) * 24 / 80
    print(f"\n{'=' * 70}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Estimated GPU-hours (H100 80GB): ~{total_gpu_hours:.0f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate scripts to reproduce paper results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Filter options
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments (default)",
    )
    parser.add_argument(
        "--main-results",
        action="store_true",
        help="Run main results only (Table 1)",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Run ablation experiments only (Table 2)",
    )
    parser.add_argument(
        "--mqar",
        action="store_true",
        help="Run MQAR experiments only (Figure 1)",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling experiments only (Figure 3)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation only (Table 3 / Figure 4)",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        action="append",
        help="Run specific experiment by name",
    )

    # Output options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--generate-script",
        action="store_true",
        help="Generate run_experiments.sh",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="run_experiments.sh",
        help="Output script path (default: run_experiments.sh)",
    )

    # Execution options
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use torchrun for multi-GPU training",
    )

    args = parser.parse_args()

    # Filter experiments
    experiments = EXPERIMENTS.copy()

    if args.experiment:
        experiments = filter_experiments(experiments, names=args.experiment)
    else:
        categories = []
        if args.main_results:
            categories.append("main")
        if args.ablations:
            categories.append("ablation")
        if args.mqar:
            categories.append("mqar")
        if args.scaling:
            categories.append("scaling")
        if args.eval:
            categories.append("eval")

        if categories:
            experiments = filter_experiments(experiments, categories=categories)

    if not experiments:
        print("No experiments selected.", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print_summary(experiments)

    # Generate or print commands
    if args.dry_run:
        print("Commands (dry-run):\n")
        for exp in experiments:
            print(f"# {exp.paper_ref}: {exp.description}")
            print(get_train_command(exp, args.multi_gpu))
            print()

    elif args.generate_script:
        script = generate_script(experiments, args.multi_gpu, args.output)

        output_path = Path(args.output)
        output_path.write_text(script)
        output_path.chmod(0o755)  # Make executable

        print(f"Generated: {args.output}")
        print(f"\nTo run: bash {args.output}")

    else:
        # Default: print summary and suggest next step
        print("Next steps:")
        print(f"  python {__file__} --dry-run       # Preview commands")
        print(f"  python {__file__} --generate-script  # Generate script")


if __name__ == "__main__":
    main()
