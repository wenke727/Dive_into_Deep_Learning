#%%
%matplotlib inline
import os
from matplotlib import animation
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn, rand
from torch.nn import functional as F
from d2l import torch as d2l

# %%

n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本


# %%

def f(x):
    return 2 * torch.sin(x) + x ** .8


y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
n_test



# %%
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
    
    
# %%

""" avg pooling """
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)


# %%
""" ⾮参数注意⼒汇聚 """
# ! Shape of `x_repeat`: (50, 50), where each row contains the same testing inputs (i.e., same queries)
x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
attention_weights = F.softmax(-(x_repeat - x_train)**2/2, dim=1)

y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

x_repeat.shape, x_train.shape, attention_weights.shape, y_train.shape, y_hat.shape


# %%
d2l.show_heatmaps(
    attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs'
)
# %%
""" 带参数注意⼒汇聚 """
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        
    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1])) # [50, 49]
        self.attention_weights = F.softmax(-((queries - keys) * self.w) ** 2, dim=1)    # [50, 49]
        
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1) # ([50, 1, 49], [50, 49, 1]  )


# %%
# `X_tile` 的形状: (`n_train`，`n_train`)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# `Y_tile` 的形状: (`n_train`，`n_train`)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# `keys` 的形状: ('n_train'，'n_train' - 1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# `values` 的形状: ('n_train'，'n_train' - 1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))


# %%
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    l = loss(net(queries=x_train, keys=keys, values=values), y_train)
    
    trainer.zero_grad()
    l.sum().backward()
    trainer.step()

    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# %%
# `keys` 的形状: (`n_test`，`n_train`)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# `value` 的形状: (`n_test`，`n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

# %%
d2l.show_heatmaps(
    net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs'
)

# %%
