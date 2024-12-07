"""
StreetCLIP evaluation from HuggingFace open-source model. 

Written by Viraj Jayam and Kapil Dheeriya, Stanford University, 3 Nov. 2024.
"""

import requests

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from transformers import CLIPProcessor, CLIPModel

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from data_loaders.streetview import StreetViewDataset

# ! Constants
NUM_TESTS = 10
BATCH_SIZE = 32

def init_model():
    model = CLIPModel.from_pretrained("vjayam07/geoguessr-clip-model")
    processor = CLIPProcessor.from_pretrained("vjayam07/geoguessr-clip-model")

    return model, processor

def hf_test(model, processor, test_df, countries):
    test_dataset = StreetViewDataset(metadata=test_df, processor=processor, return_country=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_correct = 0
    total_samples = 0

    for batch in test_dataloader:
        images = batch["image"]
        labels = batch["country"]

        choices = [f"A Street View photo in {country}." for country in countries]
        inputs = processor(text=choices, images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity score
            probs = logits_per_image.softmax(dim=1)

        predictions = torch.argmax(probs, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += len(labels)

    accuracy = total_correct / total_samples
    return accuracy

def test(model, processor, test_dfs, countries):
    accuracies = []

    for test_df in test_dfs:
        accuracy = hf_test(model, processor, test_df, countries)
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)
    return mean_accuracy, accuracies

def main():
    model, processor = init_model()

    metadata = pd.read_csv("full_data/street_view_metadata.csv")

    test_dfs = []
    for i in range(NUM_TESTS):
        _, test = train_test_split(metadata, test_size=0.1, random_state=42 + i)  # Varying random state
        test_dfs.append(test)

    countries = metadata["country"].unique()

    mean_accuracy, accuracies = test(model, processor, test_dfs, countries)

    print(f"Mean Accuracy across {NUM_TESTS} tests: {mean_accuracy:.4f}")
    for i, acc in enumerate(accuracies):
        print(f"Accuracy for test split {i + 1}: {acc:.4f}")

if __name__ == '__main__': 
    main()
