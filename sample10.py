#!/usr/bin/env python3

# For a training pipeline we have the following steps
# 1) design model (input, output size, fast forward)
# 2) construct the loss and optimizer
# 3) training loop
#    - forward pass: compute the prediction
#    - backward pass: gradients
#    - update weights

import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# The dataset has 3 different wine categories
class WineDataset(Dataset):
    # Initialize and load the dataset
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt("data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # We want to predict the first column: the wine category
        # The other columns are x
        self.x = xy[:, 1:]
        # That first column is our y
        self.y = xy[:, [0]]

        # Store transformation
        self.transform = transform


    # Returns the entry at index
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        return self.transform(sample) if self.transform else sample


    # Returns the length
    def __len__(self):
        return self.n_samples

# Transforms the numpy object to tensors
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

# Sample for multiplication
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

#tensor_transform = ToTensor()
#dataset = WineDataset(transform=tensor_transform)
#features, labels = dataset[0]
#print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
print(dataset[0])
features, labels = dataset[0]


