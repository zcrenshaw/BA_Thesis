from __future__ import print_function
from datetime import date

import numpy as np
import pandas as pd

import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as T

from VQVAE import VQ_VAE

global device

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(model,optimizer,training_loader,num_training_updates,data_variance):
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity, _ = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

    return model, train_res_recon_error, train_res_perplexity


def save(model,recon_error,perplexity,num_training_updates,data):
    '''Saving model and data'''

    today = date.today().strftime("%m-%d-%y")
    name = "{}_{}_{}".format(data,num_training_updates,today)

    # Save model
    model.eval()
    model_name = name + ".pth"
    torch.save(model.state_dict(), './models/' + model_name)

    # Save csv with stats
    stats = {
        "Update": range(num_training_updates),
        "Reconstruction Error": recon_error,
        "Perplexity": perplexity,
        }
    csv_name = name + ".csv"
    pd.DataFrame(data=stats).to_csv('./csvs/' + csv_name)

    return name


def avg_size(tensors):
    x = []
    y = []

    for t in tensors:
        x.append(t[0].size()[1])
        y.append(t[0].size()[2])

    return int(np.mean(x)), int(np.mean(y))


def test(model,data):
    model.eval()

    (valid_originals, types) = next(iter(data))
    valid_originals = valid_originals.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    return valid_quantize, types


def main():
    '''Use on Google Colab ONLY'''
    # Mount to drive

    # from google.colab import drive
    # drive.mount('/content/gdrive')

    '''Setup Model'''

    batch_size = 32
    num_training_updates = int(sys.argv[1])

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 64
    num_embeddings = 512

    commitment_cost = 0.25

    decay = 0.99

    learning_rate = 1e-3

    model = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                   num_embeddings, embedding_dim,
                   commitment_cost, decay).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    '''Data Ingestion'''

    #PATH = '/content/gdrive/My Drive/CS/CMSC-254/ASL-Handshape-Dataset/'
    root = './'
    data = 'datasets/'
    set = 'SIO-IBUSY-D'
    PATH = root + data + set + "/"


    # Get proper size

    train_pre = datasets.ImageFolder(root= PATH, transform=T.ToTensor())

    x, y = avg_size(train_pre)

    _, recon_size, _ ,_  = model(torch.rand(1,3, x,y))

    rx = recon_size.shape[2]
    ry = recon_size.shape[3]

    # Transform data

    data_transform = T.Compose([
                            T.Resize((rx,ry)),
                            T.ToTensor(),
                            T.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                            ])
    data = datasets.ImageFolder(root= PATH, transform=data_transform)

    subset = len(data) // 10
    torch.manual_seed(0)

    train_data, test_data = random_split(data, [len(data)-subset, subset])

    data_tensors = []

    for t in train_data:
        data_tensors.append(t[0]/255.0)


    data_variance = torch.var(torch.cat(data_tensors, dim=0))


    training_loader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)

    '''Train'''

    print("Training Model...")

    trained_model,recon_error,perplexity = train(model,
                                                optimizer,
                                                training_loader,
                                                num_training_updates,
                                                data_variance)
    '''Save'''

    print("Saving Model...")

    save(trained_model,recon_error,perplexity,num_training_updates,set)

    print("Done!")


if __name__ == "__main__":
    main()