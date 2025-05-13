import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader, Dataset, random_split
import torch
from model import StudentTrainer
from utils import ImgDatasetWithCovars, preprocess_img
from tqdm import tqdm
from collections import defaultdict

torch.set_float32_matmul_precision('high')

with open("cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    
student_cfg = cfg["student"]
data_cfg = cfg["data"]

data = torch.load(data_cfg['path'])
data['image'] = preprocess_img(data['image'])
assert data['image'].isnan().any() == False, "NaN values found in images"
train_dataset = ImgDatasetWithCovars(data, range(len(data['image'])))

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

model = StudentTrainer(student_cfg)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=student_cfg["max_epochs"],
    default_root_dir=student_cfg["save_dir"],
    log_every_n_steps=cfg["logging"]["log_interval"],
    enable_progress_bar=cfg["logging"]["enable_progress_bar"]
)

trainer.fit(model, train_dataloader, val_dataloader)
dat = model.extract_embeddings(val_dataloader)
torch.save(dat, "data_with_embeddings.pt")

# full_data = torch.load(data_cfg['path'])
# full_data['image'] = preprocess_img(full_data['image'])
# full_dataset = ImgDatasetWithCovars(full_data, range(len(full_data['image'])))
# full_dataloader = DataLoader(
#     full_dataset,
#     batch_size=data_cfg["batch_size"],
#     shuffle=False,
#     num_workers=data_cfg["num_workers"]
# )

# encoder = model.student

# encoder.eval()
# new_data = defaultdict(list)

# for batch in tqdm(full_dataloader):
#     images = batch['image'].to(encoder.device)
#     embeddings = encoder.embed(images)

#     for key in batch:
#         new_data[key].append(batch[key].cpu())
#     new_data['embedding'].append(embeddings.cpu())

# final_data = {k: torch.cat(v, dim=0) for k, v in new_data.items()}
# torch.save(final_data, student_cfg['save_dir'] + '/data_with_embeddings.pt')
