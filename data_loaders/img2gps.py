"""
Data loader class for the IMG2GPS(3K) datasets. 

# ! Needs more work to be done. As of 11-13, we are working with Google Streetview Images.
"""

import json
import clip
from PIL import Image
from torch.utils.data import Dataset

import clip

class Img2gpsDataset(Dataset):
    def __init__(self, image_path, list_txt):
        self.image_path = image_path
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        img = Image.open(self.image_path[idx])
        title = self.title[idx]
        return img, title