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
NUM_EPOCHS=5
LR=1e-5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def define_args(parser):
    parser.add_argument("--dataset_name", required=True, help="Path to the dataset folder.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--cluster_centers", required=True, help="Path to the metadata for cluster centers CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output.")

    return parser

class MLPPlus(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_clusters, dropout_rate=0.5):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in range(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_clusters))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
        # x should be of shape [batch_size, input_dim]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

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


def train_country(base_country, num_clusters, metadata):
    if base_country == "United_States":
        wandb.init(
            project="country_clip_training", 
            name=f"classifier_run_{datetime.datetime.now()}",
            config={
                "learning_rate": 1e-6,
                "batch_size": BATCH_SIZE,
                "epochs": 5
            }
        )
    country_metadata = metadata[metadata['country'] == base_country]

    model = MLP(input_dim=512, hidden_dim=128, num_clusters=num_clusters)
    clip_model = CLIPModel.from_pretrained("vjayam07/geoguessr-clip-model")
    processor = CLIPProcessor.from_pretrained("vjayam07/geoguessr-clip-model")
    clip_model = clip_model.to(device)
    model = model.to(device)

    train_df, _ = train_test_split(country_metadata, test_size=0.1, random_state=42)

    train_dataset = ClassificationDataset(metadata=train_df,
                                          clip_model=clip_model,
                                          processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(params=model.parameters(), 
                      lr=LR,
                      betas=(0.9, 0.98),
                      weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss().to(device)

    repo_id = f"vjayam07/{base_country}_classifier"

    model.apply(reset_weights)
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_dataloader, desc=f"Epoch #{epoch} Training"):
            images, _, truth_cluster = batch
            truth_cluster = truth_cluster.to(device)
            images = images.cuda()
            image_features = clip_model.get_image_features(pixel_values=images.squeeze(0))

            # forwardprop
            outputs = model(image_features)

            # backprop
            loss = loss_fn(outputs, truth_cluster)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if base_country == "United_States":
            wandb.log({"loss": loss.item(), "epoch": epoch})
    
    api = HfApi()
    torch.save(model.state_dict(), "mlp_model.pth")
    api.upload_file(
        path_or_fileobj="mlp_model.pth",
        path_in_repo="mlp_model.pth",
        repo_id=repo_id,
        repo_type="model"
    )

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