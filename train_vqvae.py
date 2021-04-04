from __future__ import print_function
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
    train_loss = []
    validation_loss = []
    for i in range(num_training_updates+1):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity, _ = model(data)
        recon_error = F.mse_loss(data_recon, data)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())

        if i%1000 == 0:
            print('%d iterations' % (i + 1))
            model.eval()
            v_loss = []
            for i in range(subset//32):
              (validation_data, _) = next(iter(validation_loader))
              validation_data = validation_data.to(device)
              v_vq_loss, v_data_recon, _, _ = model(validation_data)
              this_recon_error = F.mse_loss(v_data_recon, validation_data)
              this_loss = this_recon_error + v_vq_loss
              v_loss.append(this_loss.item())

            v_loss_val = np.mean(v_loss)
            validation_loss.append(v_loss_val)
            if (v_loss_val == np.min(validation_loss) and i > 0):
              save_model(model,"",i,args)

            print('loss: %.3f' % v_loss_val)
            print()
            model.train()

    return model, train_loss, validation_loss


def main():
    '''Setup Model'''

    model, optimizer, args = make_VQVAE()
    model = model.to(device)

    '''Data Ingestion'''

    batch_size = 32
    training_loader, validation_loader , _, subset = all_signers(batch_size)

    '''Train'''


    trained_model, train_loss, validation_loss= train(model,
                                                                            optimizer,
                                                                            training_loader,
                                                                            validation_loader,
                                                                            args.num_updates,
                                                                            args,
                                                                            subset)

    '''Save Data'''

    save_data(train_loss,validation_loss,args.num_updates,args)


if __name__ == "__main__":
    main()
