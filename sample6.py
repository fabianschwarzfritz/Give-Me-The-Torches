#!/usr/bin/env python3


# For a training pipeline we have the following steps
# 1) design model (input, output size, fast forward)
# 2) construct the loss and optimizer
# 3) training loop
#    - forward pass: compute the prediction
#    - backward pass: gradients
#    - update weights

# In this file we are using pytorch instead of doing the gradient manually
# with numpy

import torch 
import torch.nn as nn

# f = w * x
# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# Alternative 1: with pre set linear model
# model = nn.Linear(input_size, output_size)

# Alternative 2: with pre set linear model
class CustomLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 20

loss = nn.MSELoss() # MSE = Mean Square Error. Returns a loss function.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Decent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    # loss
    l = loss(Y, y_pred)
    # gradients - this is different with pytorch
    # This is equal to the backward pass.
    l.backward()

    # update weights
    optimizer.step()

    # We must zero the gradients again.
    optimizer.zero_grad()
    
    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training f(5) = {model(X_test).item():.3f}')

