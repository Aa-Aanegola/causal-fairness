import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader, Dataset, random_split
import torch
from model import TeacherModel
from utils import ImgDatasetWithCovars, preprocess_img

torch.set_float32_matmul_precision('high')

with open("cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)

oracle_cfg = cfg["oracle"]
data_cfg = cfg["data"]

data = torch.load(data_cfg['path'])
data['image'] = preprocess_img(data['image'])
val_size = int(len(data['image']) * data_cfg["val_ratio"])
train_size = len(data['image']) - val_size
train_indices, val_indices = torch.utils.data.random_split(range(len(data['image'])), [train_size, val_size])

train_dataset = ImgDatasetWithCovars(data, train_indices.indices)
val_dataset = ImgDatasetWithCovars(data, val_indices.indices)

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

model = TeacherModel(oracle_cfg)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=oracle_cfg["save_dir"],
    filename="teacher-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
    save_weights_only=True
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=oracle_cfg["max_epochs"],
    callbacks=[checkpoint_callback, lr_monitor],
    default_root_dir=oracle_cfg["save_dir"],
    log_every_n_steps=cfg["logging"]["log_interval"],
    enable_progress_bar=cfg["logging"]["enable_progress_bar"]
)

trainer.fit(model, train_dataloader, val_dataloader)