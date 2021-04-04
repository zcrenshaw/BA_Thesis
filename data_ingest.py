import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as T

import utils
import h5py
import numpy as np


class AllSignersDataset(Dataset):

    def __init__(self, h5_paths, transform):
        self.h5_paths = h5_paths
        self.labels = torch.empty((1,1)).float()
        self.images = torch.empty((1,1,128,128)).float()
        for p in self.h5_paths:
        	data = h5py.File(p, 'r')
		self.labels = torch.cat((self.labels,torch.from_numpy(np.array(data['L'])).float()))
		self.images = torch.cat((self.images,torch.from_numpy(np.array(data['X'])).view(-1,1,128,128).float()))
        self.length = len(self.labels)
    	self.transform = transform

    def __getitem__(self, index): #to enable indexing
	label = self.labels[index]
	
	# ignore non letter inputs, and letters J (9) and Z (25)
	while (label < 0 or label > 24 or label == 9):
		index += 1
		if index >= self.length:
			index = 0
		label = self.labels[index]

	image = self.images[index]

        image = self.transform(image)
        return (
                image,
                label,
        )

    def __len__(self):
        return self.length

# adapted from https://stackoverflow.com/questions/59190493/trouble-crating-dataset-and-dataloader-for-hdf5-file-in-pytorch-not-enough-valu

class OneSignerDataset(Dataset):

    def __init__(self, h5_path, transform):
        self.h5_path = h5_path
        self.data = h5py.File(h5_path, 'r')
	self.labels = torch.from_numpy(np.array(self.data['L'])).float()
	self.images = torch.from_numpy(np.array(self.data['X'])).view(-1,1,128,128).float()
        self.length = len(self.labels)
    	self.transform = transform

    def __getitem__(self, index): #to enable indexing
	label = self.labels[index]
	
	# ignore non letter inputs, and letters J (9) and Z (25)
	while (label < 0 or label > 24 or label == 9):
		index += 1
		if index >= self.length:
			index = 0
		label = self.labels[index]

	image = self.images[index]

        image = self.transform(image)
        return (
                image,
                label,
        )

    def __len__(self):
        return self.length

def all_signers(batch_size):

    data_root = '/share/data/asl-data/fsvid/'

    mat_paths = [ data_root + 'image128_color0_andy.mat'
                 , data_root + 'image128_color0_drucie.mat'
                 , data_root + 'image128_color0_rita.mat'
                 , data_root + 'image128_color0_robin.mat']

    data_transform = T.Compose([
                        T.Normalize((0.5,), (1.0,)),
                        ])
    #data = OneSignerDataset(mat_paths[0],data_transform)
    data = AllSignersDataset(mat_paths,data_transform)
    loaders = make_loaders(data,batch_size)
    return loaders


def make_loaders(data,batch_size):
    subset = len(data) // 10
    split = [len(data)-2*subset, subset,subset]
    generator = torch.Generator().manual_seed(0)
    train_data, validation_data, test_data = random_split(data, split)

    training_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=False,
				  num_workers=0)

    validation_loader = DataLoader(validation_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=False,
				  num_workers=0)

    test_loader = DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=False,
			          num_workers=0)

    return training_loader, validation_loader, test_loader, subset
