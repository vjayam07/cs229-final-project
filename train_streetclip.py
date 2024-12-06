'''
We train the CLIP model with StreetView images.
'''
import os
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import adamw
from torch.utils.data import DataLoader
import pandas as pd

import clip
from transformers import CLIPProcessor, CLIPModel

from tqdm import tqdm

from data_loaders.streetview import StreetViewDataset


# ! Constants
BATCH_SIZE = 16
NUM_EPOCHS=100


def define_args(parser):
    parser.add_argument("--dataset_name", required=True, help="Path to the dataset folder.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output.")

    return parser

def contrastive_loss(logits_per_image, logits_per_text):
    '''
    CLIP contrastive loss function.
    '''
    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_text, labels)
    return (image_loss + text_loss) / 2


def train(**kwargs):
    metadata = kwargs.get('metadata_file', None)
    dataset_name = kwargs.get('dataset_name', None)
    output_dir = kwargs.get('output_dir', None)

    metadata = pd.read_csv(metadata)
    metadata['filename'] = metadata['filename'].apply(lambda x: os.path.join(dataset_name, x))

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = StreetViewDataset(metadata=metadata,
                                processor=processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = adamw(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            images, texts = batch
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            texts = texts.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            outputs = model(pixel_values=images, input_ids=texts)
            logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
            
            loss = contrastive_loss(logits_per_image, logits_per_text)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = define_args(parser)

    args = parser.parse_args()
    train(**vars(args))



# ! Notes
"""
1) Code for text embeddings from CLIP
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

outputs = model(**inputs)
text_embeds = outputs.text_embeds

2) SAM ViT embeddings
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
pretrained weights for SAM ViT model^

3) Kapil will work on writing code for MLP and I will work on CLIP training and self attention.
Both on training loop

Kapil also work on the data processing loop.
"""