import numpy as np
import torch
from torch import nn
from constants import *


class InvertibleFunction:
    def __init__(self, dim=16, seed=0):
        np.random.seed(seed)

        # Random orthogonal matrices (invertible and well-conditioned)
        Q1, _ = np.linalg.qr(np.random.randn(dim, dim))
        Q2, _ = np.linalg.qr(np.random.randn(dim, dim))
        self.A = Q1
        self.B = Q2
        self.b = np.random.randn(dim)
        self.c = np.random.randn(dim)

        self.A_inv = np.linalg.inv(self.A)
        self.B_inv = np.linalg.inv(self.B)

    def forward(self, W):
        h = np.dot(self.B, W.T).T + self.b
        h = np.tanh(h)
        rep = np.dot(self.A, h.T).T + self.c
        return rep

    def inverse(self, y):
        h = np.dot(self.A_inv, (y - self.c).T).T
        h = np.arctanh(np.clip(h, -0.999999, 0.999999))  # clip for numerical stability
        W = np.dot(self.B_inv, (h - self.b).T).T
        return W
    
    
def sample_sfm_confounded(n_samples, decoder, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    U_xz = np.random.randn(n_samples, 1)
    U_x  = np.random.randn(n_samples, 1) * sigma_X
    U_z  = np.random.randn(n_samples, 1) * sigma_Z
    
    inv_f = InvertibleFunction(dim=DIM_W)

    X = (U_xz + U_x > 0).astype(float)
    Z = a * U_xz + b * U_z            

    XZ = np.concatenate([X, Z], axis=1)
    W = XZ @ A.T + np.random.randn(n_samples, DIM_W) * sigma_W
    
    W_prime = inv_f.forward(W)
    
    logits = beta_1 * X.squeeze() + beta_2 * Z.squeeze() + W @ beta_3 + np.random.randn(n_samples) * sigma_Y
    Y = (logits > 0).astype(int)
    

    W_tensor = torch.tensor(W, dtype=torch.float32)
    images = decoder(W_tensor)
    

    return {
        'X': torch.tensor(X, dtype=torch.float32),
        'Z': torch.tensor(Z, dtype=torch.float32),
        'W': W_tensor,
        'W_prime': torch.tensor(W_prime, dtype=torch.float32),
        'Y': torch.tensor(Y, dtype=torch.long),
        'image': images
    }
    
import pandas as pd

def dump_sfm_to_csv(data, csv_path):
    B, _, H, W = data['image'].shape
    flat_images = data['image'].view(B, -1).numpy()
    W_flat = data['W'].numpy()
    W_prime_flat = data['W_prime'].numpy()
    X = data['X'].numpy().squeeze()
    Z = data['Z'].numpy().squeeze()
    Y = data['Y'].numpy().squeeze()
    
    inv_f = InvertibleFunction(dim=16)
    
    

    image_cols = [f"img_{i}" for i in range(flat_images.shape[1])]
    w_cols = [f"w_{i}" for i in range(W_flat.shape[1])]
    w_prime_cols = [f"w_prime_{i}" for i in range(W_prime_flat.shape[1])]

    df = pd.DataFrame(flat_images, columns=image_cols)
    for i, col in enumerate(w_cols):
        df[col] = W_flat[:, i]
    for i, col in enumerate(w_prime_cols):
        df[col] = W_prime_flat[:, i]
    df['X'] = X
    df['Z'] = Z
    df['Y'] = Y

    df.to_csv(csv_path, index=False)
