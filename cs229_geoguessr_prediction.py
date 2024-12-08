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

# ! Constants
BATCH_SIZE = 1

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
    model = CLIPModel.from_pretrained(HF_dir)
    processor = CLIPProcessor.from_pretrained(HF_dir)

    return model, processor

# ! two coordinates to kilometers
def haversine(lat1, lon1, lat2, lon2):
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
 
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
 
    a = (pow(math.sin(dLat / 2), 2) + pow(math.sin(dLon / 2), 2) * math.cos(lat1) * math.cos(lat2))
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c

def collate_fn(batch):
    return batch[0][0], batch[0][1]

def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def id_multi_clusters(cluster_centers):
    cluster_centers = pd.read_csv(cluster_centers)
    cluster_counts = cluster_centers.groupby('Country')['Cluster'].nunique()

    countries_multiple_clusters = set(cluster_counts[cluster_counts > 1].index.tolist())
    country_counts = cluster_centers['Country'].value_counts().to_dict()
    
    final_dict = {}
    for country in country_counts:
        if country in countries_multiple_clusters:
            final_dict[country] = country_counts[country]

    return final_dict




##########

def full_test(model, processor, test_df, countries, countries_num_clusts, cluster_metadata):
    test_dataset = FinalTestDataset(metadata=test_df, processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    final_dict = {
        'city': 0, # 25km
        'region': 0, # 200km
        'country': 0, # 750km
        'continent': 0 # 2500km
    }

    data_len = test_dataset.__len__()

    model.eval()
    for images, ground_truth_coords in test_dataloader:
        choices = [f"A Street View photo in {country}." for country in countries]
        inputs = processor(text=choices, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        pred_idx = torch.argmax(probs, dim=1)
        preds = countries[pred_idx[0]]
        
        model_path = hf_hub_download(repo_id=f"vjayam07/{preds}_classifier", filename="mlp_model.pth")
        num_clusters = countries_num_clusts[preds]
        mlp = MLP(input_size=512, hidden_size=128, output_size=num_clusters)
        mlp.load_state_dict(torch.load(model_path))
        mlp.eval()

        images = images.cuda()
        image_features = model.get_image_features(pixel_values=images.squeeze(0))

        outputs = mlp(image_features)
        pred_cluster = torch.argmax(outputs, dim=1)
        coordinate_prediction = cluster_metadata.loc[(cluster_metadata['Country'] == preds) & (cluster_metadata['Cluster'] == pred_cluster)]

        distance = haversine(ground_truth_coords[0],
                             ground_truth_coords[1],
                             coordinate_prediction['Cluster_Center_Latitude'],
                             coordinate_prediction['Cluster_Center_Longitude'])

        if distance <= 25:
            city += 1 / data_len
        if distance > 25 and distance <= 200:
            region += 1 / data_len
        if distance > 200 and distance <= 750:
            country += 1 / data_len
        if distance > 750 and distance <= 2500:
            country += 1 / data_len



    return final_dict


def test_fn(**kwargs):
    metadata = kwargs.get('metadata_file', None)
    dataset_name = kwargs.get('dataset_name', None)
    HF_dir = kwargs.get('clip_dir', None)
    clusters_csv = kwargs.get('cluster_centers', None)

    clusters_metadata = pd.read_csv(clusters_csv)
    countries_num_clusts = id_multi_clusters(clusters_metadata)

    model, processor = init_model(HF_dir)
    model.to(device)

    metadata = pd.read_csv("full_data/street_view_metadata.csv")
    metadata['filename'] = metadata['filename'].apply(lambda x: os.path.join(dataset_name, x))

    # _, test_df = train_test_split(metadata, test_size=0.1, random_state=42)

    countries = metadata["country"].unique()
    final_dict = full_test(model, processor, metadata, countries, countries_num_clusts, clusters_metadata)
    print(final_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = define_args(parser)

    args = parser.parse_args()
    test_fn(**vars(args))
