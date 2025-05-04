import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import yaml

class InverseDecoder(nn.Module):
    def __init__(self, image_size=32, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class InverseDecoderModule(pl.LightningModule):
    def __init__(self, informative_dims, lr=1e-3):
        super().__init__()
        self.model = InverseDecoder()
        self.informative_dims = torch.tensor(informative_dims)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def selective_mse_loss(self, pred, target):
        mask = torch.zeros_like(target)
        mask[:, self.informative_dims] = 1.0
        diff = (pred - target) * mask
        mse = (diff ** 2).sum() / (mask.sum() + 1e-8)
        return mse

    def training_step(self, batch, batch_idx):
        imgs, w_true = batch
        w_pred = self(imgs)
        loss = self.selective_mse_loss(w_pred, w_true)
        self.log("train_mse", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def get_dataset(data, batch_size=64):
    W = data['W']
    w_mean = W.mean(dim=0, keepdim=True)
    w_std = W.std(dim=0, keepdim=True) + 1e-8  # avoid division by zero
    W_norm = (W - w_mean) / w_std

    img = data['image']
    img_mean = img.mean()
    img_std = img.std() + 1e-8
    img = (img - img_mean) / img_std

    dataset = TensorDataset(data['image'], W_norm)
    
    print(f"W Mean: {w_mean}, Std: {w_std}")
    print(f"Dataset size: {len(dataset)}")
    
    return dataset, w_mean, w_std

def dump_to_csv(data, model, csv_path):
    model.eval()
    dset, w_mean, w_std = get_dataset(data)
    preds = []
    for i in range(len(dset)):
        img, w = dset[i]
        
        w_pred = model(img.unsqueeze(0))
        w_pred = w_pred.squeeze(0) * w_std + w_mean
        preds.append(w_pred.detach().cpu().numpy())

    preds = np.array(preds).squeeze()
    df = pd.DataFrame(preds, columns=[f"w_pred_{i}" for i in range(preds.shape[1])])
    
    og_df = pd.read_csv(csv_path)
    df = pd.concat([og_df, df], axis=1)
    df.to_csv(csv_path.split('.')[0] + 'app.csv', index=False)
    
if __name__ == "__main__":
    cfg = yaml.safe_load(open('./configs/reconstruct.yaml', 'r'))
    
    data = torch.load(cfg['data_path'])
    
    full_dataset, w_mu, w_sigma = get_dataset(data)

    # 80/20 split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        logger=cfg['logger']
    )

    informative_dims = cfg['informative_dims']
    model = InverseDecoderModule(informative_dims=informative_dims)

    trainer.fit(model, train_loader)

    model.eval()
    total_mse = 0.0
    n = 0

    with torch.no_grad():
        for x, w_true in test_loader:
            x = x.to(model.device)
            w_true = w_true.to(model.device)
            w_pred = model(x)
            loss = model.selective_mse_loss(w_pred, w_true)
            total_mse += loss.item() * x.size(0)
            n += x.size(0)
            
            if n <= 256:
                print(w_true[0, torch.tensor(cfg['informative_dims'])], w_pred[0, torch.tensor(cfg['informative_dims'])])

    print(f"Test MSE (informative dims only): {total_mse / n:.6f}")
    
    dump_to_csv(data, model, cfg['csv_path'])