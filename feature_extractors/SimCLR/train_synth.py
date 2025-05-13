import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lightly.transforms.simclr_transform import SimCLRTransform
from torch.utils.data import DataLoader, Dataset, random_split
from lightly.data import LightlyDataset
import torch
from torchvision.transforms.functional import to_pil_image

import torchvision.transforms as T

from model import SimCLRModel 
# from utils import CXRSelfSupervisedDataset
from utils import SimCLRGrayscaleTransform
from utils import ImgDataset, preprocess_img

torch.set_float32_matmul_precision('high')

with open("cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)

simclr_cfg = cfg["simclr"]
data_cfg = cfg["data"]

data = torch.load(data_cfg['path'])
data['image'] = preprocess_img(data['image'])

val_size = int(len(data['image']) * data_cfg["val_ratio"])
train_size = len(data['image']) - val_size
train_indices, val_indices = torch.utils.data.random_split(range(len(data['image'])), [train_size, val_size])

train_images = data['image'][train_indices.indices]
val_images = data['image'][val_indices.indices]

train_targets = torch.arange(len(train_images))
val_targets = torch.arange(len(val_images))

transform = SimCLRGrayscaleTransform()

train_dataset = ImgDataset(train_images, train_targets, transform=transform)
val_dataset = ImgDataset(val_images, val_targets, transform=transform)

train_dataset = LightlyDataset.from_torch_dataset(train_dataset)
val_dataset = LightlyDataset.from_torch_dataset(val_dataset)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=data_cfg["batch_size"],
    shuffle=True,
    num_workers=data_cfg["num_workers"],
    drop_last=True
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=data_cfg["batch_size"],
    shuffle=False,
    num_workers=data_cfg["num_workers"],
    drop_last=False
)

model = SimCLRModel(cfg['simclr'])

checkpoint_callback = ModelCheckpoint(
    dirpath=simclr_cfg["save_dir"],
    save_top_k=1,
    monitor="val_alignment",
    mode="min",
    filename="may10-with-d",
    save_weights_only=True
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    max_epochs=simclr_cfg['optimizer']["max_epochs"],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint_callback, lr_monitor],
    default_root_dir=simclr_cfg["save_dir"],
    log_every_n_steps=cfg["logging"]["log_interval"],
    enable_progress_bar=cfg["logging"]["enable_progress_bar"]
)

trainer.fit(model, train_dataloader, val_dataloader)
