"""
StreetCLIP evaluation from HuggingFace open-source model. 

Written by Viraj Jayam and Kapil Dheeriya, Stanford University, 3 Nov. 2024.
"""

import requests
import argparse
import os

import torch
import torch.nn as nn
import pandas as pd
import math
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from transformers import CLIPProcessor, CLIPModel

from data_loaders.streetview import FinalTestDataset

from huggingface_hub import hf_hub_download
from tqdm import tqdm

import pyautogui
import yaml
import os
from time import sleep

from select_regions import get_coords
from geoguessr_bot import GeoBot

# ! Constants
BATCH_SIZE = 1

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of units in each hidden layer.
            num_clusters (int): Number of output classes.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_clusters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def define_args(parser):
    parser.add_argument("--dataset_name", required=True, help="Path to the dataset folder.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--cluster_centers", required=True, help="Path to the cluster centers metadata CSV file.")
    parser.add_argument("--clip_dir", required=True, help="Directory to save the output.")

    return parser

def init_model(HF_dir):
    hash_id = '8222552f78548103bcb07f0fd02abae172b375b0'
    model = CLIPModel.from_pretrained(HF_dir, revision=hash_id)
    processor = CLIPProcessor.from_pretrained(HF_dir, revision=hash_id)

    return model, processor

# ! two coordinates to kilometers
import math

def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    rLat1 = math.radians(lat1)
    rLon1 = math.radians(lon1)
    rLat2 = math.radians(lat2)
    rLon2 = math.radians(lon2)
    
    # Differences
    dLat = rLat2 - rLat1
    dLon = rLon2 - rLon1
    
    # Haversine formula
    a = (math.sin(dLat / 2) ** 2) + math.cos(rLat1) * math.cos(rLat2) * (math.sin(dLon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    return r * c


def collate_fn(batch):
    return batch[0][0], batch[0][1], batch[0][2], batch[0][3]

def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def id_multi_clusters(cluster_centers):
    cluster_counts = cluster_centers.groupby('Country')['Cluster'].nunique()

    countries_multiple_clusters = set(cluster_counts[cluster_counts > 1].index.tolist())
    country_counts = cluster_centers['Country'].value_counts().to_dict()
    
    final_dict = {}
    for country in country_counts:
        if country in countries_multiple_clusters:
            final_dict[country] = country_counts[country]

    return final_dict



def full_test(model, processor, images, countries, countries_num_clusts, cluster_metadata):
    model.eval()
    choices = [f"A Street View photo in {country}." for country in countries]
    
    inputs = processor(text=choices, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    pred_idx = torch.argmax(probs, dim=1)
    preds = countries[pred_idx.item()]

    print(preds)

    all_model_countries = set(countries_num_clusts.keys())
    needs_model = (preds in all_model_countries)
    if not needs_model:
        coordinate_prediction = cluster_metadata.loc[(cluster_metadata['Country'] == preds) & (cluster_metadata['Cluster'] == 0)]
        lat = coordinate_prediction['Cluster_Center_Latitude']
        long = coordinate_prediction['Cluster_Center_Longitude']
        lat = lat.iloc[0]
        long = long.iloc[0]
    else: 
        model_path = hf_hub_download(repo_id=f"vjayam07/{preds}_classifier", filename="mlp_model.pth")
        num_clusters = countries_num_clusts[preds]
        mlp = MLP(input_dim=512, hidden_dim=128, num_clusters=num_clusters)
        mlp.load_state_dict(torch.load(model_path, map_location='cpu'))
        mlp.eval()
        
        mlp.to(device)
        # images_proc = images_proc.cuda()
        image_proc = processor(images=images, return_tensors="pt")
        image_proc = image_proc['pixel_values']
        image_features = model.get_image_features(pixel_values=image_proc)

        outputs = mlp(image_features)
        pred_cluster = torch.argmax(outputs, dim=1)
        coordinate_prediction = cluster_metadata.loc[(cluster_metadata['Country'] == preds) & (cluster_metadata['Cluster'] == pred_cluster.item())]
        
        lat = coordinate_prediction['Cluster_Center_Latitude'].iloc[0]
        long = coordinate_prediction['Cluster_Center_Longitude'].iloc[0]

    
    return f"Lat: {lat}, Lon: {long}"



def play_turn(bot: GeoBot, plot: bool = False):
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)
    # screenshot_b64 = GeoBot.pil_to_base64(screenshot)
    images = screenshot

    clusters_metadata = pd.read_csv("full_data/cluster_centers.csv")
    countries_num_clusts = id_multi_clusters(clusters_metadata)
    HF_dir = "vjayam07/geoguessr-clip-model"

    model, processor = init_model(HF_dir)
    model.to(device)

    old_metadata = pd.read_csv("full_data/street_view_metadata.csv")
    countries = old_metadata["country"].unique()

    answer = full_test(model, processor, images, countries, countries_num_clusts, clusters_metadata)

    location = bot.extract_location_from_response(answer)
    
    bot.select_map_location(*location, plot=plot)
    pyautogui.press(" ")
    sleep(2)


def main(turns=5, plot=False):
    metadata=""
    if "screen_regions.yaml" not in os.listdir():
        screen_regions = get_coords(players=1)
    with open("screen_regions.yaml") as f:
        screen_regions = yaml.safe_load(f)

    bot = GeoBot(
        screen_regions=screen_regions,
        player=1
    )

    for turn in range(turns):
        print("\n----------------")
        print(f"Turn {turn+1}/{turns}")
        play_turn(bot=bot, plot=plot)


if __name__ == "__main__":
    main(turns=5, plot=True)
