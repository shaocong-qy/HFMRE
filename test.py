import torch

a=torch.rand(2,3,8)
print(a.size(2)*a.size(1))

b=a.reshape(6,8)
print(b)

c=b.reshape(2,3,8)
print(c)