""" 
Dataloader for Google StreetView images dataset. 
"""

import torch
from torch.utils.data import Dataset
from PIL import Image


class StreetViewDataset(Dataset):
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

        text_inputs = self.processor(
            text=[country_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        return image, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

class StreetViewTestDataset(Dataset):
    def __init__(self, metadata, processor):
        self.metadata = metadata
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = self.metadata.iloc[idx]['filename']
        image = Image.open(image_path).convert("RGB")
        
        country = self.metadata.iloc[idx]['country']

        return image, country

class FinalTestDataset(Dataset):
    def __init__(self, metadata, processor):
        self.metadata = metadata
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = self.metadata.iloc[idx]['filename']
        
        gt_country = self.metadata.iloc[idx]['country']

        image = Image.open(image_path).convert("RGB")
        image_proc = self.processor(images=image, return_tensors="pt")
        image_proc = image_proc['pixel_values']
        
        coords = [self.metadata.iloc[idx]['Latitude'], self.metadata.iloc[idx]['Longitude']]

        return image, image_proc, coords, gt_country
    


class ClassificationDataset(Dataset):
    def __init__(self, metadata, clip_model, processor):
        self.metadata = metadata
        self.clip_model = clip_model
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = self.metadata.iloc[idx]['filename']
        image = Image.open(image_path).convert("RGB")
        image = self.processor(images=image, return_tensors="pt")

        image = image['pixel_values']

        country = self.metadata.iloc[idx]['country']
        cluster = self.metadata.iloc[idx]['cluster']

        return image, country, cluster