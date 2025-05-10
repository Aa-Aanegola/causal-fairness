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

ckpt_path = "/insomnia001/depts/edu/COMSE6998/aa5506/causal-fairness/feature_extractors/SimCLR/ckpt/student/lightning_logs/version_15/checkpoints/epoch=9-step=3910.ckpt"
data_path = "/insomnia001/depts/edu/COMSE6998/aa5506/causal-fairness/synth_data/data/synth_data_new.pt"

st = StudentTrainer.load_from_checkpoint(ckpt_path)
encoder = st.student
encoder.eval()

full_data = torch.load(data_path)
full_data['image'] = preprocess_img(full_data['image'])
full_dataset = ImgDatasetWithCovars(full_data, range(len(full_data['image'])))
full_dataloader = DataLoader(
    full_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

new_data = defaultdict(list)

with torch.no_grad():
    for batch in tqdm(full_dataloader):
        images = batch['image'].to(encoder.device)
        embeddings = encoder.embed(images)

        for key in batch:
            new_data[key].append(batch[key].cpu())
        new_data['embedding'].append(embeddings.cpu())

final_data = {k: torch.cat(v, dim=0) for k, v in new_data.items()}
torch.save(final_data, 'data_with_embeddings.pt')