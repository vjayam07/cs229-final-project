""" 
Dataloader for Google StreetView images dataset. 
"""

import json
import clip
from PIL import Image
from torch.utils.data import Dataset

import clip

class StreetViewDataset(Dataset):
    '''
    self.metadata: pandas df. 
    self.processor: CLIP processor

    Note the above initializations!
    '''
    def __init__(self, metadata, processor):
        self.metadata = metadata
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = self.metadata.iloc[idx]['filename']
        image = Image.open(image_path).convert("RGB")
        image = self.processor(images=image, return_tensors="pt").pixel_values[0]
        
        country = self.metadata.iloc[idx]['country']
        country_text = f"A Street View photo in {country}."
        country_text = self.processor(text=country_text, return_tensors="pt", padding=True, truncation=True).input_ids[0]

        return image, country_text