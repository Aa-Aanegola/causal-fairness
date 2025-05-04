import numpy as np
import torch
from torch import nn

class ImageDecoder(nn.Module):
    def __init__(self, image_size=32, keep_layers=None):
        super().__init__()
        self.image_size = image_size
        self.xx, self.yy = torch.meshgrid(
            torch.arange(image_size), torch.arange(image_size), indexing='ij'
        )
        self.cx, self.cy = image_size // 2, image_size // 2
        
        if keep_layers is None:
            self.keep_layers = range(16)
        else:
            self.keep_layers = keep_layers

    def forward(self, W):
        """
        W: Tensor of shape (B, 16)
        Returns: Tensor of shape (B, 1, H, W)
        """
        B = W.shape[0]
        H, W_ = self.image_size, self.image_size
        xx, yy = self.xx.float(), self.yy.float()

        imgs = []
        for i in range(B):
            w = W[i]
            img = torch.zeros((H, W_))

            if 0 in self.keep_layers:
                # Circle radius
                r = ((xx - self.cx)**2 + (yy - self.cy)**2).sqrt()
                img += torch.clamp((w[0] * 5 - r), min=0, max=1)

            if 1 in self.keep_layers:
                # Square
                half = int(torch.clamp(torch.abs(w[1]) * 8, 2, 12).item())
                img[self.cy - half:self.cy + half, self.cx - half:self.cx + half] += 0.5

            if 2 in self.keep_layers:
                # Vertical stripes
                img += 0.5 * torch.sin((w[2] * 10) * torch.pi * xx / H)

            if 3 in self.keep_layers:
                # Horizontal stripes
                img += 0.5 * torch.sin((w[3] * 10) * torch.pi * yy / H)

            if 4 in self.keep_layers:
                img += 0.5 * torch.sin((w[4] * 5) * torch.pi * (xx + yy) / (2 * H))


            if 5 in self.keep_layers:
                # Sinusoidal warp
                warp = torch.sin((yy + w[5] * 10) * torch.pi / H)
                img += 0.5 * warp

            if 6 in self.keep_layers:
                # Brightness offset
                img += w[6]

            if 7 in self.keep_layers:
                # Gaussian blob
                bx = int(self.cx + w[7] * 5)
                by = int(self.cy + w[7] * 5)
                sigma = 4 + torch.abs(w[7]) * 4
                g = torch.exp(-((xx - bx)**2 + (yy - by)**2) / (2 * sigma**2))
                img += g

            if 8 in self.keep_layers:
                # XOR ring
                ring = ((r - w[8] * 10)**2 < 10).float()
                img += ring

            if 9 in self.keep_layers:
                # Checkerboard
                cb_size = int(torch.clamp(torch.abs(w[9]) * 8 + 2, 2, 16).item())
                cb = ((xx // cb_size + yy // cb_size) % 2) * 0.3
                img += cb

            if 10 in self.keep_layers:
                # Diagonal bar
                angle = w[10] * torch.pi
                rot_line = torch.cos(angle) * (xx - self.cx) + torch.sin(angle) * (yy - self.cy)
                img += ((torch.abs(rot_line) < 3).float() * 0.6)

            if 11 in self.keep_layers:
                # Radial pattern
                theta = torch.atan2(yy - self.cy, xx - self.cx)
                img += 0.5 * torch.sin(theta * w[11] * 5)

            if 12 in self.keep_layers:
                # Add noise
                img += torch.randn_like(img) * 0.1 * w[12]

            if 13 in self.keep_layers:
                # Texture grain
                grain = torch.rand_like(img)
                img += (grain - 0.5) * w[13] * 0.05

            if 14 in self.keep_layers:
                # # Contrast
                img = (img - 0.5) * (1 + w[14]) + 0.5

            if 15 in self.keep_layers:
                # Invert
                if w[15] > 0:
                    img = 1.0 - img

            # Normalize and clamp
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            imgs.append(img.unsqueeze(0))

        return torch.stack(imgs, dim=0)