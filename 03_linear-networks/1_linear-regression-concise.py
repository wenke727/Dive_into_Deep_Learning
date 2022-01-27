#%%

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

#%%
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# %%
# define model

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
print(net[0])
net[0].weight.data.normal_(0, .01)
net[0].bias.data.fill_(0)

# %%
# loss func
loss = nn.MSELoss()

# Optim
trainer = torch.optim.SGD(net.parameters(), lr=.03)

#%%
num_epoches = 3
for epoch in range(num_epoches):
    for X, y in data_iter:
        l = loss(net(X), y)
        
        trainer.zero_grad()
        l.backward()
        trainer.step()
    
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# %%
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)


# %%
