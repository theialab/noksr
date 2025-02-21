import os
import numpy as np
import torch
from torch.utils.data import Dataset
from noksr.data.dataset.scannet import Scannet
from noksr.data.dataset.synthetic import Synthetic

class Mixture(Dataset):
    def __init__(self, cfg, split):
        
        self.scannet_dataset = Scannet(cfg, split)
        self.synthetic_dataset = Synthetic(cfg, split)
        self.cfg = cfg
        self.split = split
        self.over_fitting = cfg.data.over_fitting
        
        # Combine the lengths of both datasets
        self.length = len(self.scannet_dataset) + len(self.synthetic_dataset)

    def __len__(self):
        if self.split == 'val':
            if self.cfg.data.validation_set == "ScanNet":
                return len(self.scannet_dataset)
            elif self.cfg.data.validation_set == "Synthetic":
                return len(self.synthetic_dataset)
            
        return self.length

    def __getitem__(self, idx):
        # Determine which dataset to load from based on idx
        if self.split == 'val':
            if self.cfg.data.validation_set == "ScanNet":
                return self.scannet_dataset[idx]
            elif self.cfg.data.validation_set == "Synthetic":
                return self.synthetic_dataset[idx]
            
        if idx < len(self.scannet_dataset):
            return self.scannet_dataset[idx]
        else:
            return self.synthetic_dataset[idx - len(self.scannet_dataset)]
