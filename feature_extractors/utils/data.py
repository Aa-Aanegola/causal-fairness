from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd


"""
Dataset structure: 
files/
    <partition_id>/
        <subject_id>/
            <study_id>/
                <image_id>.png
                ...
            ...
        ...
    ...
"""


class CXRDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor(), split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        
        split_annotations = os.path.join(root_dir, 'mimic-cxr-2.0.0-split.csv')
        if not os.path.exists(split_annotations):
            raise FileNotFoundError(f"Split annotations file not found at {split_annotations}")
        
        metadata = os.path.join(root_dir, 'mimic-cxr-2.0.0-metadata.csv')
        if not os.path.exists(metadata):
            raise FileNotFoundError(f"Metadata file not found at {metadata}")
        df = pd.read_csv(metadata, usecols=['dicom_id', 'ViewPosition'])
        self.view_positions = {row['dicom_id']: row['ViewPosition'] for _, row in df.iterrows()}
            
        
        with open(split_annotations, 'r') as f:
            lines = f.readlines()
            self.adm_files = [line.strip().split(',')[0] for line in lines if line.strip().split(',')[-1] == split]
            self.adm_files = set(self.adm_files)
        
        file_root_dir = os.path.join(root_dir, 'files')
        
        for partition in os.listdir(file_root_dir):
            partition_dir = os.path.join(file_root_dir, partition)
            for subject in os.listdir(partition_dir):
                subject_dir = os.path.join(partition_dir, subject)
                for study in os.listdir(subject_dir):
                    study_dir = os.path.join(subject_dir, study)
                    for image in os.listdir(study_dir):
                        if image.split('.')[0] in self.adm_files and self.view_positions[image.split('.')[0]] == 'AP':
                            image_path = os.path.join(study_dir, image)
                            self.files.append(image_path)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        partition, subject, study, image_id = image_path.split('/')[-4:]
        return {
                'image': image, 
                'partition': partition, 
                'subject': subject, 
                'study': study, 
                'dicom': image_id.split('.')[0]
            }
        
        
def get_transform():
    return transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    
    
class CXRWithXZDataSet(Dataset):
    def __init__(self, root_dir : str, annotations : str, split : str = 'train'):
        self.transform = get_transform()
        
        self.dicom_to_file = {}
        
        file_root_dir = os.path.join(root_dir, 'files')
        
        metadata = os.path.join(root_dir, 'mimic-cxr-2.0.0-metadata.csv')
        if not os.path.exists(metadata):
            raise FileNotFoundError(f"Metadata file not found at {metadata}")
        df = pd.read_csv(metadata, usecols=['dicom_id', 'ViewPosition'])
        self.view_positions = {row['dicom_id']: row['ViewPosition'] for _, row in df.iterrows()}
        
        for partition in os.listdir(file_root_dir):
            partition_dir = os.path.join(file_root_dir, partition)
            for subject in os.listdir(partition_dir):
                subject_dir = os.path.join(partition_dir, subject)
                for study in os.listdir(subject_dir):
                    study_dir = os.path.join(subject_dir, study)
                    for image in os.listdir(study_dir):
                        if image.split('.')[0] and self.view_positions[image.split('.')[0]] == 'AP':
                            image_path = os.path.join(study_dir, image)
                            self.dicom_to_file[image.split('.')[0]] = image_path
        
        self.annotations = pd.read_csv(annotations)
        self.annotations = self.annotations[self.annotations['dicom_id'].isin(self.dicom_to_file.keys())]
        self.annotations = self.annotations[self.annotations['split'] == split]
        
        self.age_mean = self.annotations['age'].mean()
        self.age_std = self.annotations['age'].std()
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        x = row['sex']
        z = (row['age'] - self.age_mean) / self.age_std
        dicom = row['dicom_id']
        image_path = self.dicom_to_file[dicom]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        partition, subject, study, image_id = image_path.split('/')[-4:]
        return {
                'image': image, 
                'partition': partition, 
                'subject': subject, 
                'study': study, 
                'dicom': image_id.split('.')[0],
                'x': x,
                'z': z,
                'y': row['respirator']
            }



class CXRSelfSupervisedDataset(Dataset):
    def __init__(self, root_dir, annotations_csv, allowed_views=['AP', 'PA']):
        self.root_dir = root_dir
        self.transform = get_transform()
        self.allowed_views = allowed_views

        self.annotations = pd.read_csv(annotations_csv)
        self.annotations = self.annotations.dropna(subset=["dicom_id", "ViewPosition"])
        self.annotations = self.annotations[self.annotations["ViewPosition"].isin(self.allowed_views)]
        self.valid_dicoms = set(self.annotations["dicom_id"].astype(str).unique())

        self.dicom_to_path = {}
        for root, _, files in os.walk(os.path.join(root_dir, "files")):
            for fname in files:
                dicom_id = fname.split('/')[-1].split('.')[0]
                if dicom_id in self.valid_dicoms:
                    self.dicom_to_path[dicom_id] = os.path.join(root, fname)

        self.image_paths = []
        for dicom_id in self.valid_dicoms:
            if dicom_id in self.dicom_to_path:
                self.image_paths.append(self.dicom_to_path[dicom_id])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, idx




if __name__ == '__main__':
    import os
    import pandas as pd
    from torchvision import transforms
    from PIL import Image
    
    root_dir = '/local/eb/aa5506/MIMIC-CXR-JPG/mimic-cxr-jpg/2.1.0/'
    annotations = '/local/eb/aa5506/MIMIC-CXR-JPG/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv'
    
    dataset = CXRSelfSupervisedDataset(root_dir, annotations)
    
    print(len(dataset))
    
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample[0].shape, sample[1])
        if i == 5:
            break