import torch
import torch.nn as nn

class ImageDecoder(nn.Module):
    def __init__(self, image_size=32, use_noise=True):
        super().__init__()
        self.image_size = image_size
        self.use_noise = use_noise
        
        # Create meshgrid
        self.xx, self.yy = torch.meshgrid(
            torch.arange(image_size), torch.arange(image_size), indexing='ij'
        )
        self.xx = self.xx.float()
        self.yy = self.yy.float()
        self.cx, self.cy = image_size // 2, image_size // 2

    def forward(self, W):
        """
        Args:
            W: Tensor of shape (B, 16)
        Returns:
            images: Tensor of shape (B, 1, H, W)
        """
        B = W.shape[0]
        H, W_ = self.image_size, self.image_size
        imgs = []

        for i in range(B):
            w = W[i]
            img = torch.zeros((H, W_))

            # Circle radius (w[0]): controls size of central circular blob
            r = ((self.xx - self.cx) ** 2 + (self.yy - self.cy) ** 2).sqrt()
            img += torch.clamp(w[0] * 6 - r, min=0, max=1)

            # Square sharpness (w[1])
            half = int(torch.clamp(torch.abs(w[1]) * 8 + 2, 2, 12).item())
            img[self.cy - half:self.cy + half, self.cx - half:self.cx + half] += 0.5

            # Vertical frequency stripes (w[2])
            img += 0.5 * torch.sin(w[2] * 2 * torch.pi * self.xx / H)

            # Horizontal frequency stripes (w[3])
            img += 0.5 * torch.sin(w[3] * 2 * torch.pi * self.yy / H)

            # Diagonal wave (w[4])
            diag = (self.xx + self.yy) / 2
            img += 0.5 * torch.sin(w[4] * torch.pi * diag / H)

            # Spatial warp (w[5])
            warp = torch.sin((self.yy + w[5] * 10) * torch.pi / H)
            img += 0.5 * warp

            # Brightness offset (w[6])
            img += w[6]

            # Gaussian blob offset (w[7])
            bx = self.cx + int(w[7] * 5)
            by = self.cy + int(w[7] * 5)
            sigma = 4 + torch.abs(w[7]) * 5
            g = torch.exp(-((self.xx - bx) ** 2 + (self.yy - by) ** 2) / (2 * sigma ** 2))
            img += g

            # XOR ring (w[8])
            ring_mask = ((r - w[8] * 10) ** 2 < 10).float()
            img += ring_mask

            # Checkerboard size (w[9])
            cb_size = int(torch.clamp(torch.abs(w[9]) * 8 + 2, 2, 16).item())
            cb = ((self.xx // cb_size + self.yy // cb_size) % 2) * 0.3
            img += cb

            # Diagonal bar (w[10])
            angle = w[10] * torch.pi
            rot_line = torch.cos(angle) * (self.xx - self.cx) + torch.sin(angle) * (self.yy - self.cy)
            img += (torch.abs(rot_line) < 3).float() * 0.6

            # Radial angle pattern (w[11])
            theta = torch.atan2(self.yy - self.cy, self.xx - self.cx)
            img += 0.5 * torch.sin(theta * w[11] * 4)

            # Optional: Add noise (w[12]) — skip if `use_noise=False`
            if self.use_noise:
                img += torch.randn_like(img) * 0.2 * w[12]

            # Optional: Texture grain (w[13])
            if self.use_noise:
                grain = torch.rand_like(img)
                img += (grain - 0.5) * w[13] * 0.2

            # Contrast adjustment (w[14])
            img = (img - 0.5) * (1 + w[14]) + 0.5

            # Invert (w[15])
            if w[15] > 0:
                img = 1.0 - img

            # Normalize
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            imgs.append(img.unsqueeze(0))

        return torch.stack(imgs, dim=0)
