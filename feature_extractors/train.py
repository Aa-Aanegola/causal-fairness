import yaml
import os
import sys
import torch
from utils import *
from models import *
from tqdm import tqdm
import pickle as pkl
from datetime import datetime
from PIL import Image
from torchvision.utils import save_image
import numpy as np

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = CXRDataset(config["data"]["root_dir"], transform=get_transform())
    print(f"Number of images in dataset: {len(dataset)}")
    test_set = CXRDataset(config["data"]["root_dir"], transform=get_transform(), split='test')
    print(f"Number of images in test set: {len(test_set)}")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"])
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    
    model = torch.compile(get_model(config["model"]))
    model = model.to(config["device"])
    
    optimizer = get_optimizer(model, config)
    criterion = get_loss(config)
    
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for i, data in enumerate(dataloader):
            image = data["image"].to(config["device"])
            optimizer.zero_grad()
            output, emb = model.forward(image)
            loss = criterion(output, image)
            loss.sum().backward()
            optimizer.step()
            print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

        # evaluate on test set
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, data in enumerate(test_dataloader):
                image = data["image"].to(config["device"])
                output, emb = model.forward(image)
                loss = criterion(output, image)
                total_loss += loss.item()
            avg_loss = total_loss / len(test_dataloader)
            
            if epoch == config["training"]["epochs"] - 1:
                # Save some images
                save_dir = os.path.join(config["output"]["img_dir"], config["model"]["name"])
                os.makedirs(save_dir, exist_ok=True)
                for i in range(4):
                    image = data["image"][i].squeeze().cpu().numpy()
                    output_image = output[i].squeeze().cpu().numpy()
                    image = (image - image.min()) / (image.max() - image.min()) * 255
                    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
                    image = image.astype(np.uint8)
                    output_image = output_image.astype(np.uint8)
                    image = Image.fromarray(image)
                    output_image = Image.fromarray(output_image)
                    image.save(os.path.join(save_dir, f"image_{i}.png"))
                    output_image.save(os.path.join(save_dir, f"output_{i}.png"))
            
            print(f"Epoch {epoch}, Test Loss: {avg_loss}")

    torch.save(model.state_dict(), os.path.join(config["output"]["ckpt_dir"], config["model"]["name"] + f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.pth"))
    
    
    embeddings = []
    
    for i, data in enumerate(dataloader):
        image = data["image"].to(config["device"])
        emb = model.encode(image).cpu().detach().numpy()
        for j in range(emb.shape[0]):
            embeddings.append({
                'partition': data['partition'][j],
                'subject': data['subject'][j],
                'study': data['study'][j],
                'dicom': data['dicom'][j],
                'embedding': emb[j]
            })
    
    with open(os.path.join(config["output"]["emb_dir"], config["model"]["name"] + f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.pkl"), "wb") as f:
        pkl.dump(embeddings, f)