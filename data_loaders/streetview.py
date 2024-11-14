""" 
Dataloader for Google StreetView images dataset. 
"""

import json
import clip
from PIL import Image
from torch.utils.data import Dataset

import clip

class StreetViewDataset(Dataset):
    def __init__(self, image_dir, list_txt):
        self.image_dir = image_dir
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        img = Image.open(self.image_path[idx])
        title = self.title[idx]
        return img, title