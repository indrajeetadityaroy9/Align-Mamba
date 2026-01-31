"""Training entry point."""

import torch
import hydra
from omegaconf import DictConfig

from align_mamba.training.distributed import setup_distributed
from align_mamba.training.trainer import NMTTrainer


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Training entry point."""
    from align_mamba.models.factory import create_model
    from align_mamba.data.loaders import create_dataloaders

    dist_info = setup_distributed()

    model = create_model(cfg, str(dist_info["device"]), torch.bfloat16)
    train_loader, val_loader = create_dataloaders(cfg, dist_info["world_size"], dist_info["rank"])

    if dist_info["is_main"]:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"model params={num_params / 1e6:.1f}M cross_attn_positions={sorted(model.decoder.cross_attn_positions)}")

    trainer = NMTTrainer(
        model=model,
        train_dataloader=train_loader,
        seed=cfg.project.seed,
        max_steps=cfg.training.max_steps,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        label_smoothing=cfg.training.label_smoothing,
        output_dir=cfg.training.output_dir,
        eval_dataloader=val_loader,
        dist_info=dist_info,
        num_samples=cfg.data.num_samples,
    )

    if cfg.training.resume_from:
        trainer.load_checkpoint(cfg.training.resume_from)

    trainer.train()


if __name__ == "__main__":
    main()
