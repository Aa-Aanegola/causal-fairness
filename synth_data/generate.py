import numpy as np
import torch
from torch import nn
from image_decoder import ImageDecoder
from utils import *
import yaml

# set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
        
if __name__ == "__main__":
    cfg = yaml.safe_load(open('./configs/generate.yaml', 'r'))
    
    decoder = ImageDecoder(image_size=cfg['image_size'])
    decoder.eval()

    n_samples = 100
    data = sample_sfm_confounded(n_samples, decoder)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(data['image'][i * 10 + j][0], cmap='gray')
            axes[i, j].axis('off')
    plt.savefig(f'{cfg["base_path"]}.png')
    
    # get 10k samples and write to CSV (no image necessary)
    import pandas as pd
    data = sample_sfm_confounded(cfg['samples'], decoder)
    
    # save to a torch .pt file
    torch.save(data, f'{cfg["base_path"]}.pt')
    
    dump_sfm_to_csv(data, f'{cfg["base_path"]}.csv')
    
    # count where X=1 and Y = 1
    df = pd.read_csv(f'{cfg["base_path"]}.csv')
    print("X=1, Y=1", len(df[(df['X'] == 1) & (df['Y'] == 1)]))
    print("X=1, Y=0",len(df[(df['X'] == 1) & (df['Y'] == 0)]))
    print("X=0, Y=1",len(df[(df['X'] == 0) & (df['Y'] == 1)]))
    print("X=0, Y=0",len(df[(df['X'] == 0) & (df['Y'] == 0)]))
    print("X=1, D=0", len(df[(df['X'] == 1) & (df['D'] == 0)]))
    print("X=0, D=1", len(df[(df['X'] == 0) & (df['D'] == 1)]))
    print("X=0, D=0", len(df[(df['X'] == 0) & (df['D'] == 0)]))
    print("X=1, D=1", len(df[(df['X'] == 1) & (df['D'] == 1)]))
    print(len(df))