#%%
%matplotlib inline
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# %%
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap="Reds"):
    """_summary_

    Args:
        matrices (_type_): values [rows, cols, queries, keys]
        xlabel (_type_): _description_
        ylabel (_type_): _description_
        titles (_type_, optional): _description_. Defaults to None.
        figsize (tuple, optional): _description_. Defaults to (2.5, 2.5).
        cmap (str, optional): _description_. Defaults to "Reds".
    """
    d2l.use_svg_display()
    
    num_rows, num_cols = matrices.shape[:2]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, 
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);


attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')

# %%
