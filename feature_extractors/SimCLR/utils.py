from torchvision import transforms
from PIL import Image
import torch

class SimCLRGrayscaleTransform:
    def __init__(self, input_size=32):
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, contrast=0.8)
            ], p=0.8),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomApply([
                transforms.RandomErasing(p=1.0, scale=(0.02, 0.2))
            ], p=0.3)
        ])

    def __call__(self, x):
        if isinstance(x, Image.Image):
            if x.mode != 'L':
                x = x.convert('L')
        elif isinstance(x, torch.Tensor):
            x = transforms.ToPILImage(mode='L')(x)
        return self.base_transform(x), self.base_transform(x)