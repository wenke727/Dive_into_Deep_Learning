#%%
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# %%
# 初始化模型参数
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=.01)
    
net.apply(init_weight)


# %%
loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=.1)

num_epochs = 10

d2l.train_ch3(net, train_iter, train_iter, loss, num_epochs, trainer)

# %%
