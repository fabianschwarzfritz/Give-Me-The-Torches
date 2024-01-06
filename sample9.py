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
    def __init__(self):
        # data loading
        xy = np.loadtxt("data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # We want to predict the first column: the wine category
        # The other columns are x
        self.x = torch.from_numpy(xy[:, 1:])
        # That first column is our y
        self.y = torch.from_numpy(xy[:, [0]])


    # Returns the entry at index
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Returns the length
    def __len__(self):
        return self.n_samples

dataset = WineDataset()

# Fun: print the data
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

batch_size = 4
num_epochs = 10
total_samples = len(dataset)
n_iterations = total_samples // batch_size + 1

print(f'samples={total_samples}, iterations={n_iterations}')

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')






