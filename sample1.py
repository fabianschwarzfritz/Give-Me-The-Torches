#!/usr/bin/env python3

import torch

x = torch.randn(3, requires_grad=True)
# pytorch will create a computational graph
print(x)

# This will create the graph
# for each operation we have nodes with inputs and outputs
# inputs: x
# inputs: 2
# operation: + 
# output: y
y = x+2

# With backpropagation we will be able to calculate the gradients
# First we do a forward pass. We apply the operation. We calculate the output y.
# As we specified it required the gradient, pytorch will automatically store
#    a function for us.  Funciton is under y.grad_fn
# This function is used in the backprop to get the gradients. 
print(y)
# prints: tensor([1.6525, 2.0769, 2.0120], grad_fn=<AddBackward0>)

z = y*y*2
print(z)
# prints: tensor([15.8259, 24.1617,  1.6347], grad_fn=<MulBackward0>)
#z = z.mean()
#print(z)

# If the output is not scalar, we need to pass in the vector we want
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# This will calculate the gradient of z wiht respect to x
# aka: dz/dx
z.backward(v)
print(x.grad)

