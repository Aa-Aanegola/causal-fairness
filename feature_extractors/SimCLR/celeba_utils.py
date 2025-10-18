import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np

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


class CelebADatasetWithCovars:
    """
    CelebA dataset class compatible with the existing framework.
    Maps CelebA causal variables to the expected format.
    """
    def __init__(self, data, indices):
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.indices = indices
        self.data = {}
        
        # Map CelebA data to expected format
        for k, v in data.items():
            self.data[k] = v[indices]
        
        # Normalize Z (age + attractiveness) if it exists
        if 'Z' in self.data:
            z_mean = self.data['Z'].mean(dim=0, keepdim=True)
            z_std = self.data['Z'].std(dim=0, keepdim=True) + 1e-8
            self.data['Z'] = (self.data['Z'] - z_mean) / z_std
    
    def __len__(self):
        return len(self.data['Y'])
    
    def __getitem__(self, index):
        img = self.transform(self.data['image'][index])
        y = self.data['Y'][index]
        x = self.data['X'][index]
        z = self.data['Z'][index]
        
        return {
            'image': img,
            'y': y,
            'x': x,
            'z': z
        }


def create_celeba_causal_data(root_dir, split='train', image_size=32):
    """
    Create CelebA data in the format expected by the causal framework.
    Returns a dictionary with the same structure as synthetic data.
    """
    from celeba_dataset import CelebACausalDataset
    
    dataset = CelebACausalDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size
    )
    
    # Convert to the expected format
    data = {
        'X': dataset.X.unsqueeze(1),  # Add dimension to match synthetic format
        'Z': dataset.Z,  # Keep 2D for age + attractiveness
        'Y': dataset.Y.unsqueeze(1),
        'image': torch.stack([dataset[i]['image'] for i in range(len(dataset))])
    }
    
    # Create dummy W and W_prime for compatibility
    # In real experiments, these would be learned features
    n_samples = len(dataset)
    data['W'] = torch.randn(n_samples, 16)  # 16D latent features
    data['W_prime'] = torch.randn(n_samples, 16)  # Transformed features
    
    return data


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
