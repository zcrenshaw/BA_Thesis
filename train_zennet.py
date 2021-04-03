from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from data_ingest import *
# ~~~~~~~~~

global device

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def test(loader,model):
  num_correct = 0
  num_samples = 0
  with torch.no_grad():
      for x, y in loader:
          x = x.to(device=device)
          y = y.to(device=device)
          output = model(x)
          _, predicted = torch.max(output, 1)
          num_samples += y.size(0)
          num_correct += (predicted == y).sum().item()

      acc = float(num_correct) / num_samples
      print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
      return acc


def train(model,optimizer,train_loader,validation_loader,num_training_updates):
  training_accuracy = []
  validation_accuracy = []
  loss_fn = nn.CrossEntropyLoss()
  model.train()
  best_model = model
  for i in range(num_training_updates):
      x, y = next(iter(train_loader))
      x = x.to(device)
      y = y.to(device)
      output = model(x)
      loss = loss_fn(output, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      training_accuracy.append(loss.item())
      if i % 100 == 0:
      	model.eval()
	val_acc = test(validation_loader, model)
        validation_accuracy.append()
        print('Iteration %d, loss = %.2f' % (i, val_acc))
        if validation_accuracy[-1] == np.min(validation_accuracy) and i > 0:
		save_model(model,"",i,args)
                best_model = model
        	model.train()
        	print()

  return best_model, training_accuracy, validation_accuracy

def main():

    print("Making model...")
    model, optimizer = make_ZenNet()
    model = model.to(device)
     
    '''Data Ingestion'''
    batch_size = 72
    num_training_updates = 5000

    print("Getting data...")
    training_loader, validation_loader , test_loader, subset = all_signers(batch_size)
    
    print ("Training model...")
    best_model, train_acc, validation_acc = train(model, optimizer, training_loader, validation_loader, num_training_updates)

    print("Testing model...")
    best_model.eval()
    test_acc = test(test_loader,best_model)
    print("Done!")

if __name__ == "__main__":
    main()
