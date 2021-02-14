import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import numpy as np
import random

class NH_HazeDataset(Dataset):

    def __init__(self, hazed_image_files, dehazed_image_files, rotation=False, color_augment=False, transform=None):

        self.hazed_image_files = [os.path.join(hazed_image_files, x) for x in sorted(os.listdir(hazed_image_files))]
        self.dehazed_image_files = [os.path.join(dehazed_image_files, x) for x in sorted(os.listdir(dehazed_image_files))]

        self.transform = transform
        self.rotation = rotation
        self.color_augment = color_augment

    def __len__(self):
        return len(self.hazed_image_files)

    def __getitem__(self, idx):
        hazed_image = Image.open(self.hazed_image_files[idx]).convert('RGB')
        dehazed_image = Image.open(self.dehazed_image_files[idx]).convert('RGB')

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
