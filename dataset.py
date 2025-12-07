import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd

class BrainMRIDataset(Dataset):
    """Dataset for brain MRI scans"""
    
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        scan_path = row['scan_path']
        label = row['label']
        
        # Load scan
        img = nib.load(scan_path)
        data = img.get_fdata()
        
        # Convert to tensor [1, 96, 96, 96]
        data = torch.FloatTensor(data).unsqueeze(0)
        
        if self.transform:
            data = self.transform(data)
        
        return data, label


class RandomFlip3D:
    """Randomly flip brain left-right"""
    def __call__(self, x):
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])
        return x