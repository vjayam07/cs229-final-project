import json
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel


def train():
    pass


def main():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    if device == "cpu":
        print("Use a GPU to train this foo")
        return

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    train()


if __name__=='__main__':
    main()