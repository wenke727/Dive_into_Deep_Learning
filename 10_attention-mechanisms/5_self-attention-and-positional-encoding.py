#%%
%matplotlib inline
import os
from numpy import reshape
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# %%
""" Self-attention """
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(
    key_size=num_hiddens, 
    query_size=num_hiddens, 
    value_size=num_hiddens, 
    num_hiddens=num_hiddens, 
    num_heads=num_heads, 
    dropout=.5)

attention.eval()

attention

nn.Linear

#%%

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens)) # [2, 4, 100]
y = attention(X, X, X, valid_lens)                     # [2, 4, 100]

X.shape, y.shape


# %%
""" Position encoding """

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)\
             / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        
        return self.dropout(X)


encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()

X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])


# %%
""" 绝对位置信息 """
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(
    P, 
    xlabel='Column (encoding dimension)',
    ylabel='Row (position)', 
    figsize=(3.5, 4), 
    cmap='Blues'
)


# %%
""" 相对位置 """

