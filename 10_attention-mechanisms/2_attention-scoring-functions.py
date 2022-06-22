#%%
%matplotlib inline
import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# %%

""" masked_softmax 测试 """

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange(
        (maxlen), 
        dtype=torch.float32,
        device=X.device
    )[None, :] < valid_len[:, None]
    X[~mask] = value
    
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行 softmax 操作"""
    # `X`: 3D张量，`valid_lens`: 1D或2D 张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        X = sequence_mask(
            X.reshape(-1, shape[-1]), 
            valid_lens,
            value=-1e6 # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        )
        return nn.functional.softmax(X.reshape(shape), dim=-1)


masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
# masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))


# %%
"""加性注意力"""
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.w_q(queries), self.w_k(keys)   # (2, 1, 20) -> (2, 1, 8); (2, 10, 2) -> (2, 10, 8)
        # sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # [2, 1, 10, 8] = [2, 1, 1, 8] + [2, 1, 10, 8]
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)             # [2, 1, 10]
        self.attention_weights = masked_softmax(scores, valid_lens) # [2, 1, 10]
        
        return torch.bmm(self.dropout(self.attention_weights), values) # [2, 1, 4] = [2, 1, 10] * [2, 10, 4]
    

# AdditiveAttention
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
y = attention(queries, keys, values, valid_lens)

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

y.shape

# %%
""" Scaled Dot-product attention """

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        return torch.bmm(self.dropout(self.attention_weights), values)


# DotProductAttention
queries = torch.normal(0, 1, (2, 1, 2))

attention = DotProductAttention(dropout=0.5)
attention.eval()

# queries, keys, values: [2, 1, 2], [2, 10, 2], [2, 10, 4]
y = attention(queries, keys, values, valid_lens) # [2, 1, 4], batch_size, queries_num, value_num

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
        
y.shape
    
# %%

# if __name__ == "__main__":


        