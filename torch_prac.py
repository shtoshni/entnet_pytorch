import torch
import torch.nn as nn
import torchtext as tt
from torchtext.data import Field, Example, Dataset

# %%
x = torch.rand(5, 3)
y = x.view(3, 5)


# %%
y.add_(1)
print (y)

if torch.cuda.is_available():
    print ("Yipree")

# %%
x = torch.rand(1)
print (x.item())

print (torch.ones_like(y))

import torch.nn

# %%
torch.manual_seed(10)
x = torch.rand(5, 3)
y = x.repeat(10, 1, 1)

#print (x)
print (y.size())
#print (z)


# %%
x = torch.rand(3, 3)
y = torch.rand(3)
y = y.unsqueeze(dim=1)

z = nn.functional.softmax(x, dim=0)
print (x)
# print (y)
print (z)


# %%

x = torch.ones(5)
y = torch.ones(5)

z = (x == y)
print (z)

# %%
