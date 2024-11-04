"""
StreetCLIP evaluation from HuggingFace open-source model. 

Written by Viraj Jayam and Kapil Dheeriya, Stanford University, 3 Nov. 2024.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import requests

import huggingface_hub as hf
from transformers import CLIPProcessor, CLIPModel
import wandb

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def init_model():
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

    return model, processor

def hf_test(model, processor):
    url = "https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)

    choices = ["San Jose", "San Diego", "Los Angeles", "Las Vegas", "San Francisco"]
    inputs = processor(text=choices, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    return probs

def main():
    model, processor = init_model()
    print(hf_test(model, processor))

if __name__ == '__main__': 
    main()
