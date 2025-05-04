import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lightly.transforms.simclr_transform import SimCLRTransform
from torch.utils.data import DataLoader

import torchvision.transforms as T

from model import SimCLRModel 
from utils import CXRSelfSupervisedDataset


with open("cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)

simclr_cfg = cfg["simclr"]
data_cfg = cfg["data"]

dataset = CXRSelfSupervisedDataset(
    root_dir=data_cfg["dataset_root"],
    annotations_csv=data_cfg["annotations_csv"],
    allowed_views=data_cfg["allowed_views"]
)

dataloader = DataLoader(
    dataset,
    batch_size=simclr_cfg["batch_size"],
    shuffle=True,
    num_workers=data_cfg["num_workers"],
    drop_last=True
)

model = SimCLRModel(cfg)

checkpoint_callback = ModelCheckpoint(
    dirpath=simclr_cfg["save_dir"],
    save_top_k=1,
    monitor="train_loss_ssl",
    mode="min",
    filename="simclr-{epoch:02d}-{train_loss_ssl:.4f}",
    save_weights_only=True
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    max_epochs=simclr_cfg["epochs"],
    accelerator="gpu" if pl.utilities.device_parser.num_gpus_available() > 0 else "cpu",
    callbacks=[checkpoint_callback, lr_monitor],
    default_root_dir=simclr_cfg["save_dir"],
    log_every_n_steps=cfg["logging"]["log_interval"]
)

trainer.fit(model, dataloader)