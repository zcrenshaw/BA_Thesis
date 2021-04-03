# Zack Crenshaw

from __future__ import print_function
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from PIL import Image

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

global alphabet
alphabet = ["a","b","c","d","e","f","g","h","i","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]

def get_data(model,data_loader,num_classes=24,num_examples=50):
    model.eval()
    model.to(device)

    encodings = defaultdict(lambda: [])
    all_examples = defaultdict(lambda: [])

    for k in range(num_classes):
      encodings[k] = []
      all_examples[k] = []

    while (keep_going(encodings,num_examples)):
      valids, types = next(iter(data_loader))
      valids = valids.to(device)
      _, _, _, this_encoding = model._vq_vae(model._pre_vq_conv(model._encoder(valids)))

      for j in range(batch_size):
        label = types[j].item()
        if label > -1 and label < 25 and label != 9:
            if len(encodings[label]) < num_examples and :
              encodings[label].append(this_encoding[j])
              all_examples[label].append(valids[j])

    encodings_tensor = torch.stack([torch.stack(encodings[n]) for n in range(num_classes)])
    all_examples_tensor = torch.stack([torch.stack(all_examples[e]) for e in all_examples.keys()])
    examples_tensor = torch.stack([all_examples[n][0] for n in range(num_classes)])

    return encodings_tensor,examples_tensor, all_examples_tensor

def nan_to_num(input,num=0.0):
  input = input.numpy()
  input = np.nan_to_num(input,num)
  return torch.tensor(input)

def entropy(encodings):
  letters, n, hw, k = encodings.shape
  # encodings: Letters x N x HW x K
  # probs: Letters x HW x K
  probs = torch.sum(encodings,dim=1)/n
  # entropy_letters: Letters x HW
  # entropy of each encoding dimension for each letter
  entropy_letters = torch.zeros((letters,hw))
  for i in range(letters):
    for j in range(hw):
      entropy_letters[i][j] = -1*torch.sum(nan_to_num(probs[i][j]*torch.log(probs[i][j])))

  # entropy_sigma: HW x 1
  # entropy of each encoding dimension across letters
  entropy_encodings = torch.zeros(hw)

  ## HW x K
  probs_hwk = torch.sum(encodings, dim=(0,1))/(n*letters)

  for m in range(hw):
    entropy_encodings[m] = -1*torch.sum(nan_to_num(probs_hwk[m]*torch.log(probs_hwk[m])))

  return entropy_letters, probs, entropy_encodings

def union(s,items):
    for i in items:
    s.add(i)
    return s

def intersect(s,items):
    if len(s) > 0:
        s = s.intersection(items)
    else:
        s = add(s,items)
    return s

def diff_candidates(entropy_letters,entropy_encodings,take,combine=union):
  vectors = set()

  # take vectors that have most entropy overall and least entropy within letter
  # as a result of overall - letter

  for i in range(len(entropy_letters)):
    diff = entropy_encodings - entropy_letters[i]
    possible = torch.argsort(diff,dim=0,descending=True)[0:take]
    # allow for union or intersection, default to union
    vectors = combine(vectors,possible.numpy())

  return vectors

def add_candidates(entropy_letters,entropy_encodings,take,combine=union):
  vectors = set()

  # take vectors that have most entropy overall and least entropy within letter
  # as a result of overall + letter

  for i in range(len(entropy_letters)):
    addition = entropy_encodings + entropy_letters[i]
    possible = torch.argsort(addition,dim=0,descending=True)[0:take]
    # allow for union or intersection, default to union
    vectors = combine(vectors,possible.numpy())

  return vectors

def similarity(encodings):
  letters, n, hw = encodings.shape
  sims = np.zeros((letters,letters))
  for x in range(letters):
    for y in range(letters):
      for i in range(hw):
        for j in range(n):
          for k in range(n):
            a = encodings[x][j][i]
            b = encodings[y][k][i]
            sims[x][y] += 1 if a == b else 0
  return sims

def numberize(tensor):
  new_tensor = torch.zeros(tensor.shape[0])
  for t in range(len(new_tensor)):
    new_tensor[t] = torch.argmax(tensor[t])
  return new_tensor

def most_likely_encoding(encodings,probs):
  letters, _, hw, k = encodings.shape
  mode = torch.zeros((letters,1,hw))
  for i in range(letters):
    for j in range(hw):
      most_likely = torch.argmax(probs[i][j])
      mode[i][0][j] = most_likely
  return mode

def distance(data,d):
  letters = data.shape[0]
  diff = np.zeros((letters,letters))
  for x in range(letters):
    for y in range(letters):
      diff[x][y] = d(data[x],data[y])
  return diff

def cosine(data):
  letters = data.shape[0]
  diff = np.zeros((letters,letters))
  for x in range(letters):
    for y in range(letters):
      diff[x][y] = torch.mean(F.cosine_similarity(all_examples[x],all_examples[y],dim=0))
  return diff

def show(img,name):
    npimg = img.numpy()
    plt.figure(figsize=(24,6))
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest',cmap='gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(name)

def show_recon(examples,model,device,name):
    show(make_grid(examples.cpu().data+0.5),"./images/{}_originals.png".format(name))

    examples = examples.to(device)
    _, examples_recon, _, _ = model(examples)
    show(make_grid(examples_recon.cpu().data+0.5),"./images/{}_recon.png".format(name))
    

def main():
    ''' Load Model and Data '''
 
    batch_size = 32
    model = load_model(device)
    _ , _ , test_loader, subset = all_signers(batch_size)

    ''' Get Encodings and Examples'''

    num_examples = 100

    encodings, examples, all_examples = get_data(model, test_loader, 24, num_examples)

    ''' Get Candidate Variables '''

    entropy_letters, probs_k, entropy_encodings = entropy(encodings.cpu())

    mode_encoding = most_likely_encoding(encodings.cpu(),probs_k)

    vectors_24 = diff_candidates(entropy_letters,entropy_encodings,1,union)
    vectors_100 = diff_candidates(entropy_letters,entropy_encodings,5,union)

    mode_24 = torch.zeros((24,1,len(vectors_24)))
    mode_100 = torch.zeros((24,1,len(vectors_100)))

    for i in range(24):
      j = 0
      for v in vectors_24:
        mode_24[i][0][j] = mode_encoding[i][0][v]
        j += 1
      j = 0
      for v in vectors_100:
        mode_100[i][0][j] = mode_encoding[i][0][v]
        j += 1

    ''' Calculate Distances '''

    image_mse = 1 - distance(all_examples,F.mse_loss)
    image_mae = 1 - distance(all_examples,F.l1_loss)
    image_cos = cosine(all_examples)

    canon = np.genfromtxt('./alphabet_features_numeric_24.csv',delimiter=',')[1::,1::]
    canon_sim = canon_similarity(canon)

    sims = similarity(mode_encoding)
    sims_24 = similarity(mode_24)
    sims_100 = similarity(mode_100)

    norm_canon = (canon_sim - 3)/8
    norm_image_mse = image_mse/np.max(image_mse)
    norm_image_mae = image_mae/np.max(image_mae)
    norm_image_cos = image_cos/np.max(image_cos)
    norm_sims = sims/np.max(sims)
    norm_sims_24 = sims_24/np.max(sims_24)
    norm_sims_100 = sims_100/np.max(sims_100)

    mse_diff = (norm_canon - (norm_image_mse))**2
    mae_diff = (norm_canon - (norm_image_mae))**2
    cos_diff = (norm_canon - (norm_image_cos))**2
    sims_diff = (norm_canon - (norm_sims))**2
    sims_24_diff = (norm_canon - (norm_sims_24))**2
    sims_100_diff = (norm_canon - (norm_sims_100))**2

    mse_diff_metric = np.sqrt(np.mean(mse_diff))
    mae_diff_metric = np.sqrt(np.mean(mae_diff))
    cos_diff_metric = np.sqrt(np.mean(cos_diff))
    sims_diff_metric = np.sqrt(np.mean(sims_diff))
    sims_24_diff_metric = np.sqrt(np.mean(sims_24_diff))
    sims_100_diff_metric = np.sqrt(np.mean(sims_100_diff))

    print(mse_diff_metric)
    print(mae_diff_metric)
    print(cos_diff_metric)
    print(sims_diff_metric)
    print(sims_24_diff_metric)
    print(sims_100_diff_metric)


if __name__ == "__main__":
    main()
