# Train Align-Mamba. Usage: python -m scripts.train --config configs/main.yaml
import argparse
import json

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from align_mamba.config import load_yaml
from align_mamba.model import HybridMambaEncoderDecoder
from align_mamba.data import create_dataloaders
from align_mamba.training import Trainer


def _emit_log(**payload) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    config, raw = load_yaml(args.config)
    resume_from = raw["training"]["resume_from"]

    set_seed(config.seed)
    accelerator = Accelerator(mixed_precision="bf16")

    model = HybridMambaEncoderDecoder(config, device=accelerator.device, dtype=getattr(torch, config.dtype))
    train_loader, val_loader = create_dataloaders(config)

    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters())
        _emit_log(
            event="train_start",
            config_path=args.config,
            params_m=round(params / 1e6, 3),
            d_state=config.d_state,
            num_pairs=config.num_pairs,
            max_steps=config.max_steps,
        )

    trainer = Trainer(model, train_loader, config, accelerator, val_loader)
    if resume_from:
        trainer.resume_from(resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
