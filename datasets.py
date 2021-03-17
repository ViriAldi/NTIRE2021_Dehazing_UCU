import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random


class NH_HazeDataset(Dataset):

    def __init__(self, hazed_image_files, dehazed_image_files, rotation=False, color_augment=False, transform=None, scale="RGB"):

        self.hazed_image_files = [os.path.join(hazed_image_files, x) for x in sorted(os.listdir(hazed_image_files))]
        self.dehazed_image_files = [os.path.join(dehazed_image_files, x) for x in sorted(os.listdir(dehazed_image_files))]

        self.transform = transform
        self.rotation = rotation
        self.color_augment = color_augment
        self.scale = scale

    def __len__(self):
        return len(self.hazed_image_files)

    def __getitem__(self, idx):
        hazed_image = Image.open(self.hazed_image_files[idx])
        dehazed_image = Image.open(self.dehazed_image_files[idx])

        if self.rotation:
            degree = random.choice([90, 180, 270])
            hazed_image = transforms.functional.rotate(hazed_image, degree) 
            dehazed_image = transforms.functional.rotate(dehazed_image, degree)

        if self.color_augment:
            hazed_image = transforms.functional.adjust_gamma(hazed_image, 1)
            dehazed_image = transforms.functional.adjust_gamma(dehazed_image, 1)                           
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            hazed_image = transforms.functional.adjust_saturation(hazed_image, sat_factor)
            dehazed_image = transforms.functional.adjust_saturation(dehazed_image, sat_factor)
            
        if self.transform:
            hazed_image = self.transform(hazed_image)
            dehazed_image = self.transform(dehazed_image)

        return {'hazed_image': hazed_image, 'dehazed_image': dehazed_image}


class MergedDataset(Dataset):

    def __init__(self, datasets):

        self.datasets = datasets

        if len({len(x) for x in datasets}) != 1:
            raise ValueError("LENGTHS DIFFER")

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):

        hazed_image = torch.cat([torch.cat([x[idx]["hazed_image"][k,:,:].unsqueeze(0) for x in self.datasets], 0) for k in range(3)], 0)
        dehazed_image = self.datasets[0][idx]["dehazed_image"]

        return {'hazed_image': hazed_image, 'dehazed_image': dehazed_image}
