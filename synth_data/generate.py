import numpy as np
import torch
from torch import nn
from decoder import Decoder
from utils import *

# set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
        
if __name__ == "__main__":
    decoder = Decoder(image_size=16, keep_layers=[0, 1, 4, 8])
    decoder.eval()

    n_samples = 100
    data = sample_sfm_confounded(n_samples, decoder)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(data['image'][i * 10 + j][0], cmap='gray')
            axes[i, j].axis('off')
    plt.savefig('synth_data_0148_16x16.png')
    
    # n_samples = 300000
    # batch_size = 10000
    # num_batches = n_samples // batch_size
    # for i in range(num_batches):
    #     data = sample_sfm_confounded(batch_size, decoder)
    #     data_list.append(data)
        
    # torch.save(data_list, f'./data/synth_data_{i}.pt')
    
    
    # get 10k samples and write to CSV (no image necessary)
    import pandas as pd
    data = sample_sfm_confounded(30000, decoder)
    
    dump_sfm_to_csv(data, 'synth_data_0148_16x16.csv')
    
    # count where X=1 and Y = 1
    df = pd.read_csv('synth_data_0148_16x16.csv')
    print("X=1, Y=1", len(df[(df['X'] == 1) & (df['Y'] == 1)]))
    print("X=1, Y=0",len(df[(df['X'] == 1) & (df['Y'] == 0)]))
    print("X=0, Y=1",len(df[(df['X'] == 0) & (df['Y'] == 1)]))
    print("X=0, Y=0",len(df[(df['X'] == 0) & (df['Y'] == 0)]))
    print(len(df))