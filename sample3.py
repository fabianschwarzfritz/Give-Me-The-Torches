#!/usr/bin/env python3

import torch

# This is about backpropagation
# Repetition about math "chain-rule" ("Kettenregel" im deutschen)

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# Forward pass:
# compute the loss
y_hat = w*x
loss = (y_hat-y)**2
print(loss)

# Backward pass:
# Pytorch will compute local gradients and backpass automatically
loss.backward()
print(w.grad)

# Update weights and next forward and backwards pass




