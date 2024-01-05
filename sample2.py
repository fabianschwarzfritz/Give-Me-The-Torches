#!/usr/bin/env python3

import torch

weights = torch.ones(4, requires_grad=True)

# sample for not updating the weights by resetting them 
#for epoch in range(3):
#    model_output = (weights * 3).sum()
#    model_output.backward()
#    print(weights.grad)
#    # This resets the weights
#    weights.grad.zero_()

# Alternative with optimizer
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
# This does the same reset of weights like above
optimizer.zero_grad()

# Recap: We should remember
# Whenever I want to calculate gradients, I need required_grads
# Can call .backward to get gradients
# If I want to do next optimization step, I must empty the gradients with .zero_()




