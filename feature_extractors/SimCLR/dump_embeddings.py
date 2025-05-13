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

ckpt_path = "/insomnia001/depts/edu/COMSE6998/aa5506/causal-fairness/feature_extractors/SimCLR/ckpt/student/lightning_logs/version_32/checkpoints/epoch=9-step=3910.ckpt"
data_path = "/insomnia001/depts/edu/COMSE6998/aa5506/causal-fairness/synth_data/data/with_d_100k.pt"

st = StudentTrainer.load_from_checkpoint(ckpt_path)

full_data = torch.load(data_path)
print(full_data.keys())
full_data['image'] = preprocess_img(full_data['image'])
assert torch.isnan(full_data['image']).any() == False, "Images should not contain NaN values"
full_dataset = ImgDatasetWithCovars(full_data, range(len(full_data['image'])))
full_dataloader = DataLoader(
    full_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

full_dat = st.extract_embeddings(full_dataloader)
torch.save(full_dat, "data_with_embeddings.pt")