import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader
import torch
from model import StudentTrainer
from celeba_utils import create_celeba_causal_data, CelebADatasetWithCovars
from tqdm import tqdm
from collections import defaultdict

torch.set_float32_matmul_precision('high')

def main():
    # Load configuration
    with open("celeba_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    student_cfg = cfg["student"]
    data_cfg = cfg["data"]

    # Create CelebA causal data
    print("Loading CelebA data...")
    data = create_celeba_causal_data(
        root_dir=data_cfg["root_dir"],
        split='train',
        image_size=data_cfg["image_size"]
    )
    
    print(f"Data loaded: {[(k, v.shape) for k, v in data.items()]}")

    # Create dataset
    train_dataset = CelebADatasetWithCovars(data, range(len(data['image'])))

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"]
    )

    val_dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"]
    )

    print(f"Total samples: {len(train_dataset)}")

    # Create student trainer model
    model = StudentTrainer(student_cfg)

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=student_cfg["max_epochs"],
        default_root_dir=student_cfg["save_dir"],
        log_every_n_steps=cfg["logging"]["log_interval"],
        enable_progress_bar=cfg["logging"]["enable_progress_bar"]
    )

    # Train the model
    print("Starting student training on CelebA...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings_data = model.extract_embeddings(val_dataloader)
    torch.save(embeddings_data, f"{student_cfg['save_dir']}/celeba_data_with_embeddings.pt")
    
    print(f"Training completed! Checkpoints and embeddings saved to: {student_cfg['save_dir']}")

if __name__ == "__main__":
    main()
