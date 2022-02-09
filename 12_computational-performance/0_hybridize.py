#%%
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

#%%
""" Sequential的混合式编程 """

def get_net():
    net = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)

# %%
# 使⽤torch.jit.script函数来转换模型，我们就有能⼒编译和优化多层感知机中的计算
net = torch.jit.script(net)
net(x)

# %%
class Benchmark:
    def __init__(self, desc=None):
        self.description = desc
    
    def __enter__(self):
        self.timer = d2l.Timer()
        return self
    
    def __exit__(self, *args):
        print(f"{self.description}: {self.timer.stop():.4f} sec")


net = get_net()
with Benchmark("without torchscript"):
    for i in range(1000): 
        net(x)

net = torch.jit.script(net)
with Benchmark('with torchscript'):
    for i in range(1000): 
        net(x)

# %%
