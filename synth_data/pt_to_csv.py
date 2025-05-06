import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

path = './data/data_with_embeddings.pt'
data = torch.load(path)


print(data.keys())

data['image'] = data['image'].numpy()
data['embedding'] = data['embedding'].numpy()
data['Y'] = data['y'].numpy()
data['X'] = data['x'].numpy()
data['Z'] = data['z'].numpy()
data['W'] = data['w'].numpy()
data['W_prime'] = data['w_prime'].numpy()

# Convert the data to a pandas DataFrame where we split all image, emb, w, w_prime into separate columns
def flatten_to_columns(key, arr):
    """Takes a [N, D] or [N, H, W] array and returns a dict of {key_i: arr[:, i]}."""
    if arr.ndim == 1:
        return {key: arr}
    else:
        N = arr.shape[0]
        flat = arr.reshape(N, -1)
        return {f"{key}_{i}": flat[:, i] for i in range(flat.shape[1])}

flat_dict = {}
for key in ['image', 'embedding', 'W', 'W_prime', 'Y', 'X', 'Z']:
    flat_dict.update(flatten_to_columns(key, data[key]))

df = pd.DataFrame(flat_dict)

df.to_csv('./data/data_with_embeddings.csv', index=False)
