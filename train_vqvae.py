from datetime import date
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
from data_ingest import all_signers
from utils import *


# ~~~~~~~~~~~

global device

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train(model,optimizer,training_loader,validation_loader,num_training_updates,args,subset):
    model.train()
    train_recon_error = []
    train_perplexity = []
    validation_recon_error = []
    validation_perplexity = []

    for i in range(num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity, _ = model(data)
        recon_error = F.mse_loss(data_recon, data)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_recon_error.append(recon_error.item())
        train_perplexity.append(perplexity.item())

        if i%1000 == 0:
            # print('%d iterations' % (i + 1))
            model.eval()
            v_recon_error = []
            v_perplexity = []
            for i in range(subset//32):
              (validation_data, _) = next(iter(validation_loader))
              _, v_data_recon, v_perplexity_i, _ = model(data)
              validation_data = validation_data.to(device)
              v_recon_error.append(F.mse_loss(v_data_recon, validation_data).item())
              v_perplexity.append(v_perplexity_i.item())

            v_recon_error_val = np.mean(v_recon_error)
            v_perplexity_val = np.mean(v_perplexity)
            validation_recon_error.append(v_recon_error_val)
            validation_perplexity.append(v_perplexity_val)
            if (v_recon_error_val == np.min(validation_recon_error) and i > 0):
              save_model(model,dataset,i,args)

            # print('recon_error: %.3f' % v_recon_error_val)
            # print('perplexity: %.3f' % v_perplexity_val)
            # print()
            model.train()

    return model, train_recon_error, train_perplexity, validation_recon_error, validation_perplexity


def main():
    '''Setup Model'''

    model, optimizer, args = make_VQVAE()
    model = model.to(device)

    '''Data Ingestion'''

    training_loader, validation_loader , _, subset = all_signers()

    '''Train'''


    trained_model, recon_error, perplexity, v_recon_error, v_perplexity = train(model,
                                                                            optimizer,
                                                                            training_loader,
                                                                            validation_loader,
                                                                            args.num_updates,
                                                                            args,
                                                                            subset)

    '''Save Data'''

    save_data(recon_error, perplexity, v_recon_error, v_perplexity)


if __name__ == "__main__":
    main()
