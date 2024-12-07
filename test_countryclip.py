"""
StreetCLIP evaluation from HuggingFace open-source model. 

Written by Viraj Jayam and Kapil Dheeriya, Stanford University, 3 Nov. 2024.
"""

import requests
import argparse
import os

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from transformers import CLIPProcessor, CLIPModel

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from data_loaders.streetview import StreetViewTestDataset

# ! Constants
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def define_args(parser):
    parser.add_argument("--dataset_name", required=True, help="Path to the dataset folder.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--HF_dir", required=True, help="Directory to save the output.")

    return parser

def init_model(HF_dir):
    model = CLIPModel.from_pretrained(HF_dir)
    processor = CLIPProcessor.from_pretrained(HF_dir)

    return model, processor

def collate_fn(batch):
    return batch[0][0], batch[0][1]

def hf_test(model, processor, test_df, countries):
    test_dataset = StreetViewTestDataset(metadata=test_df, processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    total_correct = 0
    total_samples = 0

    model.eval()
    for images, truth_countries in test_dataloader:
        choices = [f"A Street View photo in {country}." for country in countries]
        inputs = processor(text=choices, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        preds = torch.argmax(probs, dim=1)
        print(preds)
        print(truth_countries)
        total_correct += (truth_countries == preds)
        total_samples += len(images)

    accuracy = total_correct / total_samples
    return accuracy


def test_fn(**kwargs):
    metadata = kwargs.get('metadata_file', None)
    dataset_name = kwargs.get('dataset_name', None)
    HF_dir = kwargs.get('HF_dir', None)

    model, processor = init_model(HF_dir)
    model.to(device)

    metadata = pd.read_csv("full_data/street_view_metadata.csv")
    metadata['filename'] = metadata['filename'].apply(lambda x: os.path.join(dataset_name, x))

    _, test_df = train_test_split(metadata, test_size=0.1, random_state=42)
    countries = metadata["country"].unique()
    accuracy = hf_test(model, processor, test_df, countries)
    
    print(f"Test accuracy: {accuracy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = define_args(parser)

    args = parser.parse_args()
    test_fn(**vars(args))
