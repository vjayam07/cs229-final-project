"""
StreetCLIP evaluation from HuggingFace open-source model. 

Written by Viraj Jayam and Kapil Dheeriya, Stanford University, 3 Nov. 2024.
"""

import numpy as np
import torch
import torch.nn as nn

import huggingface_hub
import wandb

