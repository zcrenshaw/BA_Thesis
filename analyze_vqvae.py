# Zack Crenshaw

from __future__ import print_function
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from PIL import Image

import umap
import os
import sys
import time
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from VQVAE import *

global device

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_data(model,data_loader,num_classes=24,num_examples=50):
    model.eval()
    model.to(device)

    encodings = defaultdict(lambda: [])
    all_examples = defaultdict(lambda: [])

    for k in range(num_classes):
      encodings[k] = []
      all_examples[k] = []

    d = nn.Upsample(1)

    while (keep_going(encodings,num_examples)):
      valids, types = next(iter(data_loader))
      valids = valids.to(device)
      _, quantized, _, this_encoding = model._vq_vae(model._pre_vq_conv(model._encoder(valids)))

      for j in range(batch_size):
        if len(encodings[types[j].item()]) < num_examples:
          class_type = types[j].item()
          encodings[class_type].append(this_encoding[j])
          all_examples[class_type].append(valids[j])

    encodings_tensor = torch.stack([torch.stack(encodings[n]) for n in range(num_classes)])
    all_examples_tensor = torch.stack([torch.stack(all_examples[e]) for e in all_examples.keys()])
    examples_tensor = torch.stack([all_examples[n][0] for n in range(num_classes)])

    return encodings_tensor,examples_tensor, all_examples_tensor

def main():
    ''' Load Model and Data '''

    model = load_model(device)
    _ , _ , test_loader, subset = all_signers()

    ''' Get Encodings and Examples'''

    encodings, examples, all_examples = get_data(model, test_loader, 24, subset)

    ''' Get Candidate Variables '''

    ''' Calculate Distances '''


if __name__ == "__main__":
    main()
