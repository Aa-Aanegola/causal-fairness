from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

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
    

def preprocess_img(img):
    img_mean = img.mean()
    img_std = img.std() + 1e-8
    img = (img - img_mean) / img_std
    return img

class ImgDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = to_pil_image(self.data[index])
        if self.transform:
            x = self.transform(x)
        
        y = self.target[index]
            
        return x, y

class ImgDatasetWithCovars(Dataset):
    def __init__(self, data, indices):
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.indices = indices
        self.data = {}
        
        for k, v in data.items():
            self.data[k] = v[indices]
        
        z_mean = self.data['Z'].mean()
        z_std = self.data['Z'].std() + 1e-8
        self.data['Z'] = (self.data['Z'] - z_mean) / z_std
    
    def __len__(self):
        return len(self.data['Y'])

    def __getitem__(self, index):
        img = self.transform(self.data['image'][index])
        y = self.data['Y'][index]
        x = self.data['X'][index]
        z = self.data['Z'][index]
        w = self.data['W'][index]
        w_prime = self.data['W_prime'][index]
        d = self.data['D'][index]
        
        return {
            'image': img,
            'y': y,
            'x': x,
            'z': z,
            'w': w,
            'w_prime': w_prime,
            'd': d
        }