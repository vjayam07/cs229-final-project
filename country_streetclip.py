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