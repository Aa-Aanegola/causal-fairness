import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader, random_split
import torch
from model import TeacherModel
from celeba_utils import create_celeba_causal_data, CelebADatasetWithCovars

torch.set_float32_matmul_precision('high')

def main():
    # Load configuration
    with open("celeba_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    oracle_cfg = cfg["oracle"]
    data_cfg = cfg["data"]

    # Create CelebA causal data
    print("Loading CelebA data...")
    data = create_celeba_causal_data(
        root_dir=data_cfg["root_dir"],
        split='train',
        image_size=data_cfg["image_size"]
    )
    
    print(f"Data loaded: {[(k, v.shape) for k, v in data.items()]}")

    # Create train/val split
    val_size = int(len(data['image']) * data_cfg["val_ratio"])
    train_size = len(data['image']) - val_size
    train_indices, val_indices = random_split(range(len(data['image'])), [train_size, val_size])

    # Create datasets
    train_dataset = CelebADatasetWithCovars(data, train_indices.indices)
    val_dataset = CelebADatasetWithCovars(data, val_indices.indices)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"]
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"]
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create teacher model
    model = TeacherModel(oracle_cfg)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=oracle_cfg["save_dir"],
        filename="teacher-celeba",
        save_top_k=1,
        mode="min",
        save_weights_only=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=oracle_cfg["max_epochs"],
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=oracle_cfg["save_dir"],
        log_every_n_steps=cfg["logging"]["log_interval"],
        enable_progress_bar=cfg["logging"]["enable_progress_bar"]
    )

    # Train the model
    print("Starting teacher training on CelebA...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    print(f"Training completed! Checkpoints saved to: {oracle_cfg['save_dir']}")

if __name__ == "__main__":
    main()
