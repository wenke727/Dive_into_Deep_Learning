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
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w

# %%
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
    
# %%
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])


# %%
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
# %%
