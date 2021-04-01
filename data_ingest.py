import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as T

import utils
import hd5py

def AllSignersDataset(Dataset):
    def __init__(self, mat_paths,transform):
        self.images = torch.empty((128,128,1))
        self.labels = torch.empty((1,1))

        for path in (mat_paths):
            data = h5py.File(path,'r')
            self.images = torch.cat(self.images,torch.from_numpy(data['X']))
            self.labels = torch.cat(self.images,torch.from_numpy(data['L']))

        # BHWC -> BCHW
        self.images = transform(self.images.permute((0,3, 1, 2)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        i = self.images[idx]
        l = self.labels[idx]
        return i,l

def OneSignerDataset(Dataset):
    def __init__(self, mat_path):
        data = h5py.File(mat_path,'r')
        self.images = torch.from_numpy(data['X'])
        self.labels = torch.from_numpy(data['L'])

        # BHWC -> BCHW
        self.images = self.images.permute((0,3, 1, 2))

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        i = self.images[idx]
        l = self.labels[idx]
        return i,l

def all_signers():

    data_root = '/share/data/asl-data/fsvid/'

    mat_paths = [ data_root + 'image128_color0_andy.mat'
                 , data_root + 'image128_color0_drucie.mat'
                 , data_root + 'image128_color0_rita.mat'
                 , data_root + 'image128_color0_robin.mat']

    data_transform = T.Compose([
                        T.Normalize((0.5), (1.0)),
                        ])

    data = AllSignersDataset(mat_paths,data_transform)

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
