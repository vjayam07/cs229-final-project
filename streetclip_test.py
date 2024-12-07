"""
StreetCLIP evaluation from HuggingFace open-source model. 

Written by Viraj Jayam and Kapil Dheeriya, Stanford University, 3 Nov. 2024.
"""

import requests
import torch
import pandas as pd

from transformers import CLIPProcessor, CLIPModel

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def init_model():
    model = CLIPModel.from_pretrained("vjayam07/geoguessr-clip-model")
    processor = CLIPProcessor.from_pretrained("vjayam07/geoguessr-clip-model")

    return model, processor

def hf_test(model, processor, countries):
    url = "https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)

    choices = [f"A Street View photo in {country}." for country in countries]
    inputs = processor(text=choices, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    probs_by_country = {countries[i]: probs[i] for i in range(len(countries))}

    pred = countries[torch.argmax(probs)]

    return probs_by_country, pred

def main():
    model, processor = init_model()

    metadata = pd.read_csv("full_data/street_view_metadata.csv")
    countries = metadata["country"].unique()

    probs, pred = hf_test(model, processor, countries)
    print(probs)
    print(f"Prediction: {pred}")
    print(f"Ground truth: United States")

if __name__ == '__main__': 
    main()
