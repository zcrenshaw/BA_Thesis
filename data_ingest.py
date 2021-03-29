import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as T

import utils

def all_signers():

    data_root = '/share/data/asl-data/fsvid/'
    folder = ''
    dataset = ''
    data_path = data_root + folder + dataset

    data_transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize((0.5), (1.0)),
                        ])
    data = datasets.ImageFolder(root= path, transform=data_transform)

    return make_loaders(data)



def make_loaders(data):
    subset = len(data) // 10
    split = [len(data)-2*subset, subset,subset]
    generator = torch.Generator().manual_seed(0)
    train_data, validation_data, test_data = random_split(data, split, generator=generator)


    training_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)

    validation_loader = DataLoader(validation_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True)

    test_loader = DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True)

    return training_loader, validation_loader, test_loader, subset
