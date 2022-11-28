"""
Written by KrishPro @ KP

filename: `data.py`
"""

from pytorch_lightning import LightningDataModule

import torch.utils.data as data
import torchvision.transforms as T
import pandas as pd
import PIL.Image
import torch
import ast
import os

data_dir = "/home/krish/Datasets/flickr30k"

class Dataset(data.Dataset):
    def __init__(self, images_dir: str, data_path: str) -> None:
        super().__init__()

        self.images_dir = images_dir

        self.transforms = T.Compose([
                T.Lambda(lambda x: x.resize((256, 256))),
                T.ToTensor(),
        ])

        self.data = pd.read_csv(data_path)
    
    def __getitem__(self, idx):
        image_name, caption = self.data.iloc[idx]
        
        image = PIL.Image.open(os.path.join(self.images_dir, image_name))

        image: torch.Tensor = self.transforms(image)

        return image, ast.literal_eval(caption)
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def collate_fn(data):
        images, captions = zip(*data)
        
        images = torch.stack(images)

        max_len = max(map(len, captions))
        captions = torch.tensor([c + ([0]*(max_len - len(c))) for c in captions])

        return images, captions

class DataModule(LightningDataModule):
    def __init__(self, images_dir: str, data_path: str, batch_size: int, val_ratio:float=0.1) -> None:
        super().__init__()

        self.images_dir = images_dir
        self.batch_size = batch_size
        self.data_path = data_path
        self.val_ratio = val_ratio

    def setup(self, stage: str=None):
        dataset = Dataset(self.images_dir, self.data_path)

        val_size = int(len(dataset) * self.val_ratio)
        
        self.train_dataset, self.val_dataset = data.random_split(dataset, [len(dataset) - val_size, val_size])

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, self.batch_size, shuffle=True, collate_fn=Dataset.collate_fn, pin_memory=True, num_workers=os.cpu_count() // 2)
    
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, self.batch_size, shuffle=True, collate_fn=Dataset.collate_fn, pin_memory=True, num_workers=os.cpu_count() // 2)