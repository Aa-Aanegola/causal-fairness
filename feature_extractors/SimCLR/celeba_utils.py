import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from celeba_dataset import CelebACausalDataset


def preprocess_celeba_img(img):
    """
    Preprocess CelebA images for training.
    Similar to the original preprocess_img but handles RGB.
    """
    if img.dim() == 4:  # Batch of images
        img_mean = img.mean(dim=(1, 2, 3), keepdim=True)
        img_std = img.std(dim=(1, 2, 3), keepdim=True) + 1e-8
    else:  # Single image
        img_mean = img.mean()
        img_std = img.std() + 1e-8
    
    img = (img - img_mean) / img_std
    return img


class CelebATransform:
    """
    CelebA-specific transform for contrastive learning.
    Handles RGB images instead of grayscale.
    """
    def __init__(self, input_size=32):
        self.base_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomResizedCrop(input_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ], p=0.8),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomApply([
                transforms.RandomErasing(p=1.0, scale=(0.02, 0.2))
            ], p=0.3)
        ])

    def __call__(self, x):
        if isinstance(x, Image.Image):
            if x.mode != 'RGB':
                x = x.convert('RGB')
        elif isinstance(x, torch.Tensor):
            if x.shape[0] == 1:  # Grayscale
                x = x.repeat(3, 1, 1)  # Convert to RGB
            x = transforms.ToPILImage()(x)
        
        return self.base_transform(x), self.base_transform(x)

def create_celeba_dataloaders(root_dir, batch_size=128, num_workers=8, image_size=32, val_ratio=0.2):
    """
    Create standard train/val dataloaders for causal training.
    """
    dataset = CelebACausalDataset(root_dir=root_dir, split="train", image_size=image_size)
    n_total = len(dataset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader


def create_celeba_causal_data(root_dir, split="train", image_size=32):
    """
    Return a dictionary version (for backward compatibility with synthetic setup).
    Still lazy — images load on-demand via the underlying Dataset.
    """
    dataset = CelebACausalDataset(root_dir=root_dir, split=split, image_size=image_size)
    return {
        "X": (dataset.celeba.attr[:, dataset.idx["male"]] + 1) / 2,
        "Z": torch.stack([
            (dataset.celeba.attr[:, dataset.idx["young"]] + 1) / 2,
            (dataset.celeba.attr[:, dataset.idx["attractive"]] + 1) / 2,
        ], dim=1),
        "Y": (dataset.celeba.attr[:, dataset.idx["smiling"]] + 1) / 2,
        "dataset": dataset,
    }


class CelebADatasetWithCovars(torch.utils.data.Dataset):
    """
    Lightweight wrapper to split CelebA data into train/val partitions.
    """
    def __init__(self, data, indices):
        base = data['dataset']
        self.images = base.image_cache
        self.X = base.X[indices]
        self.Z = base.Z[indices]
        self.Y = base.Y[indices]
        self.indices = indices
        self.base = base

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        if self.images is not None:
            img = self.images[idx]
        else:
            img = self.base.__getitem__(idx)['image']
        return {
            'image': img,
            'x': self.X[i],
            'z': self.Z[i],
            'y': self.Y[i]
        }


def analyze_celeba_causal_relationships(data):
    """
    Analyze causal relationships in CelebA data.
    """
    X = data['X'].squeeze()
    Z_age = data['Z'][:, 0]
    Z_attractive = data['Z'][:, 1]
    Y = data['Y'].squeeze()
    
    print("CelebA Causal Analysis:")
    print(f"Total samples: {len(X)}")
    print(f"Gender distribution (X): {X.mean():.3f} (0=Male, 1=Female)")
    print(f"Age mean (Z_age): {Z_age.mean():.3f} (0=Old, 1=Young)")
    print(f"Attractiveness mean (Z_attractive): {Z_attractive.mean():.3f}")
    print(f"Smile rate (Y): {Y.mean():.3f}")
    
    print("\nCorrelations:")
    print(f"X-Y (Gender-Smile): {torch.corrcoef(torch.stack([X, Y]))[0, 1]:.3f}")
    print(f"Z_age-Y (Age-Smile): {torch.corrcoef(torch.stack([Z_age, Y]))[0, 1]:.3f}")
    print(f"Z_attractive-Y (Attractiveness-Smile): {torch.corrcoef(torch.stack([Z_attractive, Y]))[0, 1]:.3f}")
    
    # Conditional statistics
    print("\nConditional Statistics:")
    male_mask = X == 0
    female_mask = X == 1
    
    print(f"Smile rate - Male: {Y[male_mask].mean():.3f}, Female: {Y[female_mask].mean():.3f}")
    print(f"Age - Male: {Z_age[male_mask].mean():.3f}, Female: {Z_age[female_mask].mean():.3f}")
    print(f"Attractiveness - Male: {Z_attractive[male_mask].mean():.3f}, Female: {Z_attractive[female_mask].mean():.3f}")


if __name__ == "__main__":
    # Test the utilities
    root_dir = "/Users/aa-aanegola/Documents/f25/research/ci2/causal-fairness/data/celeba_dataset"
    
    print("Testing CelebA utilities...")
    data = create_celeba_causal_data(root_dir, split='train')
    print(f"Data keys: {data.keys()}")
    print(f"Data shapes: {[(k, v.shape) for k, v in data.items()]}")
    
    analyze_celeba_causal_relationships(data)
