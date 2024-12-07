""" 
Dataloader for Google StreetView images dataset. 
"""

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
