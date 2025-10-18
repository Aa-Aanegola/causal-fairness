import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
import numpy as np
from PIL import Image
import os

class CelebACausalDataset(Dataset):
    """
    CelebA dataset with causal variable mapping for fairness experiments.
    
    Causal Variables:
    - X (Treatment): Gender (Male=0, Female=1)
    - Z (Confounder): Attractiveness (continuous, derived from Attractive attribute)
    - D (Decision): Smile Detection (binary, from Smiling attribute)
    - Y (Outcome): Eyeglasses (binary, from Eyeglasses attribute)
    """
    
    def __init__(self, root_dir, split='train', image_size=32, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Load CelebA dataset
        # Use the parent directory as root since torchvision expects specific structure
        celeba_root = os.path.dirname(root_dir)
        self.celeba_dataset = CelebA(
            root=celeba_root,
            split=split,
            target_type='attr',
            transform=None,  # We'll apply transforms manually
            download=False
        )
        
        # Define attribute indices
        self.attr_names = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
            'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
            'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young'
        ]
        
        # Get indices for our causal variables
        self.male_idx = self.attr_names.index('Male')
        self.attractive_idx = self.attr_names.index('Attractive')
        self.young_idx = self.attr_names.index('Young')
        self.smiling_idx = self.attr_names.index('Smiling')
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB normalization
            ])
        else:
            self.transform = transform
            
        # Pre-compute causal variables
        self._precompute_causal_variables()
    
    def _precompute_causal_variables(self):
        """Pre-compute all causal variables for efficiency."""
        n_samples = len(self.celeba_dataset)
        
        self.X = torch.zeros(n_samples)  # Gender: Male=0, Female=1
        self.Z = torch.zeros(n_samples, 2)  # Age + Attractiveness (2D continuous)
        self.Y = torch.zeros(n_samples)   # Smile Detection
        
        for i in range(n_samples):
            _, attributes = self.celeba_dataset[i]
            
            # X: Gender (Male=0, Female=1)
            # CelebA Male attribute: -1=male, 1=female
            self.X[i] = 1 if attributes[self.male_idx] == 1 else 0
            
            # Z: Age + Attractiveness (2D continuous)
            # Age: Young=1, Old=0 (map from [-1,1] to [0,1])
            age = (attributes[self.young_idx] + 1) / 2
            # Attractiveness: map from [-1,1] to [0,1]
            attractiveness = (attributes[self.attractive_idx] + 1) / 2
            self.Z[i] = torch.tensor([age, attractiveness])
            
            # Y: Smile Detection (binary)
            self.Y[i] = 1 if attributes[self.smiling_idx] == 1 else 0
    
    def __len__(self):
        return len(self.celeba_dataset)
    
    def __getitem__(self, index):
        image, _ = self.celeba_dataset[index]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'x': self.X[index],
            'z': self.Z[index],
            'y': self.Y[index]
        }
    
    def get_causal_stats(self):
        """Get statistics about causal variables."""
        stats = {
            'n_samples': len(self),
            'X_mean': self.X.mean().item(),
            'X_std': self.X.std().item(),
            'Z_age_mean': self.Z[:, 0].mean().item(),
            'Z_age_std': self.Z[:, 0].std().item(),
            'Z_attractive_mean': self.Z[:, 1].mean().item(),
            'Z_attractive_std': self.Z[:, 1].std().item(),
            'Y_mean': self.Y.mean().item(),
            'X_Y_correlation': torch.corrcoef(torch.stack([self.X, self.Y]))[0, 1].item(),
            'Z_age_Y_correlation': torch.corrcoef(torch.stack([self.Z[:, 0], self.Y]))[0, 1].item(),
            'Z_attractive_Y_correlation': torch.corrcoef(torch.stack([self.Z[:, 1], self.Y]))[0, 1].item(),
        }
        return stats


class CelebASimCLRDataset(Dataset):
    """
    CelebA dataset for SimCLR training (self-supervised learning).
    Returns augmented image pairs for contrastive learning.
    """
    
    def __init__(self, root_dir, split='train', image_size=32):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Load CelebA dataset
        # Use the parent directory as root since torchvision expects specific structure
        celeba_root = os.path.dirname(root_dir)
        self.celeba_dataset = CelebA(
            root=celeba_root,
            split=split,
            target_type='attr',
            transform=None,
            download=False
        )
        
        # SimCLR-style transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
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
    
    def __len__(self):
        return len(self.celeba_dataset)
    
    def __getitem__(self, index):
        image, _ = self.celeba_dataset[index]
        
        # Apply same transform twice to get two augmented views
        view1 = self.transform(image)
        view2 = self.transform(image)
        
        return (view1, view2), torch.tensor(index), torch.tensor(index)


def create_celeba_dataloaders(root_dir, batch_size=256, num_workers=8, image_size=32):
    """
    Create train/val dataloaders for CelebA causal experiments.
    """
    # Create datasets
    train_dataset = CelebACausalDataset(
        root_dir=root_dir,
        split='train',
        image_size=image_size
    )
    
    val_dataset = CelebACausalDataset(
        root_dir=root_dir,
        split='valid',
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.get_causal_stats()


def create_celeba_simclr_dataloaders(root_dir, batch_size=256, num_workers=8, image_size=32):
    """
    Create train/val dataloaders for CelebA SimCLR training.
    """
    # Create datasets
    train_dataset = CelebASimCLRDataset(
        root_dir=root_dir,
        split='train',
        image_size=image_size
    )
    
    val_dataset = CelebASimCLRDataset(
        root_dir=root_dir,
        split='valid',
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    root_dir = "/Users/aa-aanegola/Documents/f25/research/ci2/causal-fairness/data/celeba_dataset"
    
    print("Testing CelebA Causal Dataset...")
    dataset = CelebACausalDataset(root_dir=root_dir, split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"X (Gender): {sample['x']}")
    print(f"Z (Age, Attractiveness): {sample['z']}")
    print(f"Y (Smile): {sample['y']}")
    
    # Print causal statistics
    stats = dataset.get_causal_stats()
    print("\nCausal Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
