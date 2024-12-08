import os
import argparse
import wandb
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

import clip
from transformers import CLIPProcessor, CLIPModel

from tqdm import tqdm

from data_loaders.streetview import ClassificationDataset

# ! Constants
BATCH_SIZE=1
NUM_EPOCHS=15
LR=1e-5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def define_args(parser):
    parser.add_argument("--dataset_name", required=True, help="Path to the dataset folder.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--cluster_centers", required=True, help="Path to the metadata for cluster centers CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output.")

    return parser

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_clusters)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


'''
# ! Make sure init_hf_classifiers has already initialized the huggingface repo for each country.
'''
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


def train_country(country, num_clusters, metadata):
    if country == "United_States":
        wandb.init(
            project="country_clip_training", 
            name=f"classifier_run_{datetime.datetime.now()}",
            config={
                "learning_rate": 1e-6,
                "batch_size": 1,
                "epochs": 10
            }
        )
    country_metadata = metadata[metadata['country'] == country]

    model = MLP(input_dim=100, hidden_dim=100, num_clusters=num_clusters)
    clip_model = CLIPModel.from_pretrained("vjayam07/geoguessr-clip-model")
    processor = CLIPProcessor.from_pretrained("vjayam07/geoguessr-clip-model")
    clip_model = clip_model.to(device)
    model = model.to(device)

    train_df, _ = train_test_split(country_metadata, test_size=0.1, random_state=42)

    train_dataset = ClassificationDataset(metadata=train_df,
                                          clip_model=clip_model,
                                          processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    print(train_dataset.__getitem__(1)['image'].size())

    optimizer = AdamW(params=model.parameters(), 
                      lr=LR,
                      betas=(0.9, 0.98),
                      weight_decay=1e-4)

    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_dataloader, desc=f"Epoch #{epoch} Training"):
            images, country, _ = batch
            images = images.to(device)

            # forwardprop
            outputs = MLP(images)
            logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text

            # backprop
            loss = torch.binary_cross_entropy_with_logits(logits_per_image, logits_per_text)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if country == "United_States":
            wandb.log({"loss": loss.item(), "epoch": epoch})

    repo_id = "vjayam07/{country}_classifier"
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)

    model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)

def train(**kwargs):
    metadata = kwargs.get('metadata_file', None)
    dataset_name = kwargs.get('dataset_name', None)
    cluster_centers = kwargs.get('cluster_centers', None)
    output_dir = kwargs.get('output_dir', None)

    metadata = pd.read_csv(metadata)
    metadata['filename'] = metadata['filename'].apply(lambda x: os.path.join(dataset_name, x))

    multi_countries = id_multi_clusters(cluster_centers)

    for country, num_clusters in multi_countries.items():
        print("#####################################################")
        print(f"STARTING TRAINING FOR {country}.")
        train_country(country, num_clusters, metadata)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = define_args(parser)

    args = parser.parse_args()
    train(**vars(args))