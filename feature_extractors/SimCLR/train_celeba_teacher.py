# train_celeba_teacher.py
import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import torch

from model import TeacherModel
from celeba_utils import create_celeba_dataloaders

torch.set_float32_matmul_precision("high")


def main():
    with open("celeba_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    oracle_cfg = cfg["oracle"]
    data_cfg = cfg["data"]

    root_dir = data_cfg["root_dir"]
    print(f"[INFO] Using CelebA root: {root_dir}")
    print(f"[INFO] CWD: {os.getcwd()}")

    print("[INFO] Creating dataloaders...")
    train_loader, val_loader = create_celeba_dataloaders(
        root_dir=root_dir,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        image_size=data_cfg["image_size"],
        val_ratio=data_cfg["val_ratio"],
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches: {len(val_loader)}")

    model = TeacherModel(oracle_cfg)

    os.makedirs(oracle_cfg["save_dir"], exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=oracle_cfg["save_dir"],
        filename="teacher-celeba",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=oracle_cfg["max_epochs"],
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=oracle_cfg["save_dir"],
        log_every_n_steps=cfg["logging"]["log_interval"],
        enable_progress_bar=cfg["logging"]["enable_progress_bar"],
    )

    print("[INFO] Starting teacher training on CelebA...")
    trainer.fit(model, train_loader, val_loader)
    print(f"[DONE] Training completed. Checkpoints in {oracle_cfg['save_dir']}")


if __name__ == "__main__":
    main()
