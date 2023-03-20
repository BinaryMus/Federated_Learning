import torch

x = torch.randn(3,4,5)
print(x)

idx = 1

tmp = x.shape

x = x.view(-1)
print(x)

x = x.view(tmp)

print(x)