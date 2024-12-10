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




##########

# def full_test(model, processor, test_df, countries, countries_num_clusts, cluster_metadata):
#     test_dataset = FinalTestDataset(metadata=test_df, processor=processor)
#     test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#     final_dict = {
#         'city': 0, # 25km
#         'region': 0, # 200km
#         'country': 0, # 750km
#         'continent': 0 # 2500km
#     }

#     data_len = test_dataset.__len__()

#     model.eval()
#     for images, proc_images, ground_truth_coords in tqdm(test_dataloader):
#         choices = [f"A Street View photo in {country}." for country in countries]
#         inputs = processor(text=choices, images=images, return_tensors="pt", padding=True)
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits_per_image = outputs.logits_per_image
#             probs = logits_per_image.softmax(dim=1)

#         pred_idx = torch.argmax(probs, dim=1)
#         preds = countries[pred_idx[0]]
        
#         model_path = hf_hub_download(repo_id=f"vjayam07/{preds}_classifier", filename="mlp_model.pth")
#         num_clusters = countries_num_clusts[preds]
#         mlp = MLP(input_size=512, hidden_size=128, output_size=num_clusters)
#         mlp.load_state_dict(torch.load(model_path))
#         mlp.eval()

#         proc_images = proc_images.cuda()
#         image_features = model.get_image_features(pixel_values=proc_images.squeeze(0))

#         outputs = mlp(image_features)
#         pred_cluster = torch.argmax(outputs, dim=1)
#         coordinate_prediction = cluster_metadata.loc[(cluster_metadata['Country'] == preds) & (cluster_metadata['Cluster'] == pred_cluster)]

#         distance = haversine(ground_truth_coords[0],
#                              ground_truth_coords[1],
#                              coordinate_prediction['Cluster_Center_Latitude'],
#                              coordinate_prediction['Cluster_Center_Longitude'])

#         if distance <= 25:
#             final_dict['city'] += 1 / data_len
#         if distance > 25 and distance <= 200:
#             final_dict['region'] += 1 / data_len
#         if distance > 200 and distance <= 750:
#             final_dict['country'] += 1 / data_len
#         if distance > 750 and distance <= 2500:
#             final_dict['continent'] += 1 / data_len



#     return final_dict

def full_test(model, processor, test_df, countries, countries_num_clusts, cluster_metadata):
    test_dataset = FinalTestDataset(metadata=test_df, processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    final_dict = {
        'city': 0, # 25km
        'region': 0, # 200km
        'country': 0, # 750km
        'continent': 0 # 2500km
    }

    total_samples = 0
    dists = 0

    model.eval()
    for images, images_proc, ground_truth_coords, gt_country in tqdm(test_dataloader):
        choices = [f"A Street View photo in {country}." for country in countries]
        inputs = processor(text=choices, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        pred_idx = torch.argmax(probs, dim=1)
        preds = countries[pred_idx.item()]

        all_model_countries = set(countries_num_clusts.keys())
        needs_model = (preds in all_model_countries)
        if not needs_model:
            coordinate_prediction = cluster_metadata.loc[(cluster_metadata['Country'] == preds) & (cluster_metadata['Cluster'] == 0)]
            lat = coordinate_prediction['Cluster_Center_Latitude']
            long = coordinate_prediction['Cluster_Center_Longitude']
        else: 
            model_path = hf_hub_download(repo_id=f"vjayam07/{preds}_classifier", filename="mlp_model.pth")
            num_clusters = countries_num_clusts[preds]
            mlp = MLP(input_dim=512, hidden_dim=128, num_clusters=num_clusters)
            mlp.load_state_dict(torch.load(model_path))
            mlp.eval()
            
            mlp.to(device)
            images_proc = images_proc.cuda()
            image_features = model.get_image_features(pixel_values=images_proc)

            outputs = mlp(image_features)
            pred_cluster = torch.argmax(outputs, dim=1)
            coordinate_prediction = cluster_metadata.loc[(cluster_metadata['Country'] == preds) & (cluster_metadata['Cluster'] == pred_cluster.item())]
            lat = coordinate_prediction['Cluster_Center_Latitude'].iloc[0]
            long = coordinate_prediction['Cluster_Center_Longitude'].iloc[0]

        distance = haversine(ground_truth_coords[0],
                             ground_truth_coords[1],
                             lat,
                             long)

        if distance <= 25:
            final_dict['city'] += 1
        if distance > 25 and distance <= 200:
            final_dict['region'] += 1
        if distance > 200 and distance <= 750:
            final_dict['country'] += 1
        if distance > 750 and distance <= 2500:
            final_dict['continent'] += 1

        total_samples += 1
        dists += 4999.91*math.pow(0.998036, distance)

    for key in final_dict.keys():
        final_dict[key] = final_dict[key] / total_samples
    
    print(f"AVERAGE GeoGuessr Score: {dists / total_samples}")
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

    metadata = pd.read_csv(metadata)
    old_metadata = pd.read_csv("full_data/street_view_metadata.csv")
    metadata['filename'] = metadata['filename'].apply(lambda x: os.path.join(dataset_name, x))

    _, test_df = train_test_split(metadata, test_size=0.1, random_state=42)

    countries = old_metadata["country"].unique()
    final_dict = full_test(model, processor, test_df, countries, countries_num_clusts, clusters_metadata)
    print(final_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = define_args(parser)

    args = parser.parse_args()
    test_fn(**vars(args))
