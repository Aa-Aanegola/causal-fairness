import os, yaml, pickle as pkl
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from PIL import Image

from utils import *                
from models import get_model    

torch.set_float32_matmul_precision("high")

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def save_samples(x, x_hat, out_dir, n=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, x.size(0))):
        fn_in  = out_dir / f"image_{i}.png"
        fn_out = out_dir / f"recon_{i}.png"
        save_image(x[i],     fn_in,  normalize=True)
        save_image(x_hat[i], fn_out, normalize=True)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))

    device  = torch.device(cfg["device"])
    beta    = cfg["model"].get("beta", 1e-3) 
    bs      = cfg["data"]["batch_size"]
    workers = cfg["data"]["num_workers"]

    ds_train = CXRDataset(cfg["data"]["root_dir"], transform=get_transform())
    ds_test  = CXRDataset(cfg["data"]["root_dir"], transform=get_transform(), split="test")
    print(f"Train: {len(ds_train)}  |  Test: {len(ds_test)}")

    dl_train = torch.utils.data.DataLoader(ds_train, bs, shuffle=True,  num_workers=workers)
    dl_test  = torch.utils.data.DataLoader(ds_test,  bs, shuffle=False, num_workers=workers)

    model = torch.compile(get_model(cfg)).to(device)
    optim = get_optimizer(model, cfg)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch:02d}")
        for batch in pbar:
            img = batch["image"].to(device)

            optim.zero_grad()
            x_hat, mu, logvar = model(img)
            loss_rec = F.mse_loss(x_hat, img, reduction="mean")
            loss_kl  = kl_divergence(mu, logvar)
            loss     = loss_rec + beta * loss_kl
            loss.backward()
            optim.step()

            pbar.set_postfix(rec=loss_rec.item(), kl=loss_kl.item())

        model.eval(); tot_rec = tot_kl = 0.0
        with torch.no_grad():
            for batch in dl_test:
                img = batch["image"].to(device)
                x_hat, mu, logvar = model(img)
                tot_rec += F.mse_loss(x_hat, img, reduction="mean").item()
                tot_kl  += kl_divergence(mu, logvar).item()

            rec_mean = tot_rec / len(dl_test)
            kl_mean  = tot_kl  / len(dl_test)
            print(f"[val] recon={rec_mean:.4f}  kl={kl_mean:.4f}")

            if epoch == cfg["training"]["epochs"] - 1:
                save_dir = Path(cfg["output"]["img_dir"]) / cfg["model"]["name"]
                save_samples(img.cpu(), x_hat.cpu(), save_dir, n=4)

    ckpt_path = Path(cfg["output"]["ckpt_dir"])
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    torch.save(model.state_dict(), ckpt_path / f"{cfg['model']['name']}_{ts}.pth")

    emb_list = []
    model.eval()
    for batch in tqdm(dl_train, desc="embeddings"):
        img = batch["image"].to(device)
        mu  = model.encode(img).cpu().numpy() 
        for j in range(mu.shape[0]):
            emb_list.append({
                "partition": batch["partition"][j],
                "subject":   batch["subject"][j],
                "study":     batch["study"][j],
                "dicom":     batch["dicom"][j],
                "embedding": mu[j]
            })

    emb_dir = Path(cfg["output"]["emb_dir"]); emb_dir.mkdir(parents=True, exist_ok=True)
    with open(emb_dir / f"{cfg['model']['name']}_{ts}.pkl", "wb") as f:
        pkl.dump(emb_list, f)

    print("Finished training & embedding extraction.")
