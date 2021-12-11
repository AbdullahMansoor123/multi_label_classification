import numpy as np
import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class multi_label_dataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir + '\\images', self.labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.labels.iloc[idx, 2:]
        label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)

        return image.float(), label.float()

#####Check dataset loader######
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = multi_label_dataset(csv_file='multi_label.csv',
                              data_dir='household/',
                              transform=transform)

print(dataset[1])