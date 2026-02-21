# Evaluate Align-Mamba. Usage: python -m scripts.eval --config configs/main.yaml
import argparse
import json
import shutil
from pathlib import Path

import torch

from align_mamba.config import load_yaml
from align_mamba.model import load_checkpoint
from align_mamba.evaluation import run_evaluation


def _emit_log(**payload) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config, raw = load_yaml(args.config)
    ckpt_cfg = raw["checkpoint"]
    eval_cfg = raw["evaluation"]

    model = load_checkpoint(ckpt_cfg["path"], config, dtype=getattr(torch, config.dtype))

    results = run_evaluation(model, config, eval_cfg)

    if results["mode"] == "capacity_cliff":
        _emit_log(
            event="eval_complete",
            mode="capacity_cliff",
            d_state=config.d_state,
            cliff_point=results["cliff_point"],
            sweep_points=len(results["results"]),
        )
    else:
        _emit_log(
            event="eval_complete",
            mode="standard",
            token_accuracy=round(results["token_accuracy"], 6),
            perplexity=round(results["perplexity"], 6),
            d_state=config.d_state,
            num_pairs=results["num_pairs"],
        )

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    shutil.copy2(args.config, out / "config.yaml")
    _emit_log(event="artifact_written", path=str(out / "metrics.json"))


if __name__ == "__main__":
    main()
