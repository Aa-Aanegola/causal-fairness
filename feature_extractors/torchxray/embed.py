import os,sys
sys.path.insert(0,"..")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dataset import MIMIC_Dataset
import pickle as pkl
from tqdm import tqdm

device = 'cuda'
torch.cuda.is_available()

import torchxrayvision as xrv
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

d_mimic_chex = MIMIC_Dataset(#datadir="/lustre03/project/6008064/jpcohen/MIMICCXR-2.0/files",
          imgpath="/local/eb/aa5506/MIMIC-CXR-JPG/mimic-cxr-jpg/2.1.0/files",
          csvpath="/local/eb/aa5506/MIMIC-CXR-JPG/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv",
          metacsvpath="/local/eb/aa5506/MIMIC-CXR-JPG/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv",
          views=["PA","AP"], unique_patients=False,
          transform=transform)

model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")

model = model.to(device)
model.eval()

dataloader = torch.utils.data.DataLoader(d_mimic_chex, batch_size=64, shuffle=False, num_workers=8)

def get_features(model, dataloader):
    features = []
    with torch.no_grad():
        try:
            for i, data in enumerate(dataloader):
                inputs = data['img'].to(device)
                outputs = model.features2(inputs)
                
                # split the batch up
                for j in range(outputs.shape[0]):
                    features.append({
                        'emb': outputs[j].cpu().numpy(),
                        'subject': data['subject_id'][j],
                        'study': data['study_id'][j],
                        'dicom': data['dicom_id'][j],
                    })
        except:
            print(len(features))
    with open('torchxray-resnet-features.pkl', 'wb') as f:
        pkl.dump(features, f)
        
get_features(model, dataloader)