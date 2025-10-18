import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader
import torch
from model import SimCLRModel 
from celeba_dataset import create_celeba_simclr_dataloaders

torch.set_float32_matmul_precision('high')

def main():
    # Load configuration
    with open("celeba_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    simclr_cfg = cfg["simclr"]
    data_cfg = cfg["data"]

    # Create CelebA dataloaders for SimCLR
    train_dataloader, val_dataloader = create_celeba_simclr_dataloaders(
        root_dir=data_cfg["root_dir"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        image_size=data_cfg["image_size"]
    )

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")

    # Create SimCLR model
    model = SimCLRModel(simclr_cfg)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=simclr_cfg["save_dir"],
        save_top_k=1,
        monitor="train_loss_ssl",
        mode="min",
        filename="simclr-celeba-{epoch:02d}-{train_loss_ssl:.4f}",
        save_weights_only=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=simclr_cfg["optimizer"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=simclr_cfg["save_dir"],
        log_every_n_steps=cfg["logging"]["log_interval"],
        enable_progress_bar=cfg["logging"]["enable_progress_bar"]
    )

    # Train the model
    print("Starting SimCLR training on CelebA...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    print(f"Training completed! Checkpoints saved to: {simclr_cfg['save_dir']}")

if __name__ == "__main__":
    main()
