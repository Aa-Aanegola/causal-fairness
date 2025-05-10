import numpy as np
import torch
from torch import nn
from constants import *
import pandas as pd


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
        h = np.arctanh(np.clip(h, -0.999999, 0.999999))
        W = np.dot(self.B_inv, (h - self.b).T).T
        return W

def structured_decision(images, X, Z, threshold=0.5):
    import torch.nn.functional as F
    from scipy.ndimage import sobel
    import numpy as np

    B = images.shape[0]
    flat_img = images.view(B, IMG_SHAPE[0], IMG_SHAPE[1]).cpu().numpy()

    brightness, noise_level, edge_strength, central_intensity, radial_symmetry = [], [], [], [], []

    for img in flat_img:
        brightness.append(img.mean())
        noise_level.append(img.std())
        sobel_x = sobel(img, axis=0)
        sobel_y = sobel(img, axis=1)
        edge_strength.append(np.sqrt(sobel_x**2 + sobel_y**2).mean())
        central = img[14:18, 14:18]
        central_intensity.append(central.mean())
        fft_var = np.var(np.fft.fftshift(np.fft.fft2(img)).real)
        radial_symmetry.append(fft_var)

    brightness = torch.tensor(brightness)
    noise_level = torch.tensor(noise_level)
    edge_strength = torch.tensor(edge_strength)
    central_intensity = torch.tensor(central_intensity)
    radial_symmetry = torch.tensor(radial_symmetry)
    radial_symmetry = (radial_symmetry - radial_symmetry.mean()) / (radial_symmetry.std() + 1e-8)

    # Normalized shape proxies
    shape_signal = edge_strength + central_intensity
    circle_strength = edge_strength
    square_strength = central_intensity
    diagonal_pattern = edge_strength * 0.8
    

    score = (
        d_w['shape_signal'] * shape_signal +
        d_w['circle_strength'] * circle_strength +
        d_w['square_strength'] * square_strength +
        d_w['diagonal_pattern'] * diagonal_pattern +
        d_w['radial_symmetry'] * radial_symmetry +
        d_w['brightness'] * brightness +
        d_w['noise'] * noise_level +
        d_w['vertical_contrast'] * edge_strength +
        d_w['central_intensity'] * central_intensity +
        d_w['x'] * X.view(-1) +
        d_w['z'] * Z.view(-1) +
        (1 - X.view(-1)) * d_w['x0']
    )

    probs = torch.sigmoid(score)
    D = (probs > threshold).int().view(-1, 1)
    return D, score


    
    
def sample_sfm_confounded(n_samples, decoder, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    U_xz = np.random.randn(n_samples, 1)
    U_x = np.random.randn(n_samples, 1) * sigma_X
    U_z = np.random.randn(n_samples, 1) * sigma_Z

    X = (U_xz + U_x > 0).astype(float)
    Z = a * U_xz + b * U_z
    XZ = np.concatenate([X, Z, 1-X], axis=1)
    W = XZ @ A + np.random.randn(n_samples, DIM_W) * sigma_W
    W_tensor = torch.tensor(W, dtype=torch.float32)
    images = decoder(W_tensor)

    # Decision based on image + features
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Z_tensor = torch.tensor(Z, dtype=torch.float32)
    D, _ = structured_decision(images, X_tensor, Z_tensor, d_threshold)
    
    # Final outcome Y
    logits = (
        (1 - X_tensor).squeeze() * (W_tensor @ torch.tensor(beta_3_low, dtype=torch.float32) + gamma_0) +
        X_tensor.squeeze() * (W_tensor @ torch.tensor(beta_3_high, dtype=torch.float32) + gamma_1) +
        beta_D * D.squeeze() +
        beta_2 * Z_tensor.squeeze() +
        torch.randn(n_samples) * sigma_Y
    )
    Y = (logits > 0).int()

    invertible_func = InvertibleFunction(dim=DIM_W)
    W_prime = invertible_func.forward(W_tensor.numpy())
    W_prime_tensor = torch.tensor(W_prime, dtype=torch.float32).view(W_tensor.shape[0], -1)

    return {
        'X': X_tensor,
        'Z': Z_tensor,
        'W': W_tensor,
        'W_prime': W_prime_tensor,
        'Y': Y,
        'D': D,
        'image': images
    }

def dump_sfm_to_csv(data, csv_path):
    B, _, H, W = data['image'].shape
    flat_images = data['image'].view(B, -1).numpy()
    W_flat = data['W'].numpy()
    W_prime_flat = data['W_prime'].numpy()
    X = data['X'].numpy().squeeze()
    Z = data['Z'].numpy().squeeze()
    Y = data['Y'].numpy().squeeze()
    D = data['D'].numpy().squeeze()
    
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
    df['D'] = D
    
    df.to_csv(csv_path, index=False)
