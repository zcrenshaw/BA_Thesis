# Utils for BA Thesis
# Zack Crenshaw

import numpy as np
import torch.nn.functional as F
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import os
import csv
import sys
import time
import random
import math
import argparse
from math import sqrt
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

from vqvae import VQ_VAE

class SquarePad:
  def __call__(self, image):
    _, w, h = image.size()
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, mode='constant',value=0)

def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-L","--num_latents", type=int
                        , choices=[32,16,8,4,2]
                        , default=32
                        , help="number of latent variables, LxL")
    parser.add_argument("-D", "--dim_embedding", type=int
                        , choices=[128,64,32,16,8,4,2]
                        , default=64
                        , help="dimension of each latent vector")
    parser.add_argument("-K", "--num_embedding", type=int
                        , choices=[1024,512,256,128,64,32,24]
                        , default=512
                        , help="number of choices in the embedding")
    parser.add_argument("-FC", "--linear"
                        , action="store_true"
                        , help="add fully connected layer before VQ")
    parser.add_argument("-n", "--num_updates", type=int
                        , default=20000
                        , help="number of training updates")

    return parser


def save_data(model,recon_error,perplexity,v_recon_error, v_perplexity, num_training_updates,args):
    L = args.num_latents
    D = args.dim_embedding
    K = args.num_embedding
    FC = "FC" if args.linear else ""
    name = "L{}_D{}_K{}_{}_{}".format(L,D,K,FC,num_training_updates)

    # Save csv with stats
    stats = {
        "Update": range(num_training_updates),
        "Reconstruction Error": recon_error,
        "Perplexity": perplexity,
        }
    csv_name = name + ".csv"
    pd.DataFrame(data=stats).to_csv('./csvs/' + csv_name)

    v_name = "L{}_D{}_K{}_{}_{}_validation".format(L,D,K,FC,num_training_updates)

    v_stats = {
        "Update": range(len(v_perplexity)),
        "Reconstruction Error": v_recon_error,
        "Perplexity": v_perplexity,
        }
    v_csv_name = v_name + ".csv"
    pd.DataFrame(data=v_stats).to_csv('./csvs/' + v_csv_name)

def save_model(model,dataset,iterations,args):
    L = args.num_latents
    D = args.dim_embedding
    K = args.num_embedding
    FC = "FC" if args.linear else ""
    name = "L{}_D{}_K{}_{}_{}".format(L,D,K,FC,iterations)

    # Save model
    model.eval()
    model_name = name +".pth"
    torch.save(model.state_dict(), './models/' + model_name)

def update_stats(args,recon_error,encoding_error):
    L = args.num_latents
    D = args.dim_embedding
    K = args.num_embedding
    FC = args.linear

    with open(root+"vqvae_stats.csv", mode="a") as stats:
        stats_writer = csv.writer(stats, delimiter=',')
        stats_writer.writerow([L,D,K,FC,recon_error,encoding])

def make_VQVAE():

    args = make_parser().parse_args()

    batch_size = 32
    num_training_updates = args.num_updates
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = args.dim_embedding
    num_embeddings = args.num_embedding
    fc = args.linear

    # shrink latent space 32x32 by 2**shrink_factor
    shrink_factor = (32 // args.num_latents) - 1

    model = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                   num_embeddings, embedding_dim,
                   commitment_cost, decay, shrink_factor, fc)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    return model, optimizer, args

def load_model(device):
    args = make_parser().parse_args()
    L = args.num_latents
    D = args.dim_embedding
    K = args.num_embedding
    i = args.num_updates

    return load_model(L,D,K,i,device)


def load_model(L,D,K,FC,i,device):
    batch_size = 32
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = D
    num_embeddings = K
    fc = FC

    # shrink latent space 32x32 by 2**shrink_factor
    shrink_factor = (32 // L) - 1

    model = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                num_embeddings, embedding_dim,
                commitment_cost, decay,shrink_factor,fc).to(device)

    FC_label = "FC" if FC else ""
    location = "./models/L{}_D{}_K{}_{}_{}.pth".format(L,D,K,FC_label,i)

    pretrained_model.load_state_dict(torch.load(location,map_location=torch.device(device)))
    pretrained_model.eval()

    return model
