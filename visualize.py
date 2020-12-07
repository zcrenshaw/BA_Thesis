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


def visualize():
    pass
