# train_celeba_student.py
import yaml
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from model import StudentTrainer
from celeba_utils import create_celeba_dataloaders

torch.set_float32_matmul_precision("high")


def main():
    with open("celeba_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    student_cfg = cfg["student"]
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

    model = StudentTrainer(student_cfg)

    os.makedirs(student_cfg["save_dir"], exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=student_cfg["save_dir"],
        filename="student-celeba",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=student_cfg["max_epochs"],
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=student_cfg["save_dir"],
        log_every_n_steps=cfg["logging"]["log_interval"],
        enable_progress_bar=cfg["logging"]["enable_progress_bar"],
    )

    print("[INFO] Starting student training on CelebA...")
    trainer.fit(model, train_loader, val_loader)
    print(f"[DONE] Training completed. Checkpoints in {student_cfg['save_dir']}")

    print("[INFO] Extracting embeddings from validation set...")
    model.eval()
    embeddings = model.extract_embeddings(val_loader)
    embeddings_path = os.path.join(student_cfg["save_dir"], "celeba_embeddings.pt")
    torch.save(embeddings, embeddings_path)
    print(f"[DONE] Embeddings saved to {embeddings_path}")


if __name__ == "__main__":
    main()
