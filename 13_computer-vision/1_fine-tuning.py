#%%
%matplotlib inline
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

from pathlib import Path

# %%
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = Path(d2l.download_extract('hotdog'))

# %%
train_imgs = torchvision.datasets.ImageFolder(data_dir / 'train')
test_imgs  = torchvision.datasets.ImageFolder(data_dir / 'test')

# %%
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);

# %%
normalize_ = torchvision.transforms.Normalize(
    [.485, .456, .406], [.229, .224, .225]
)

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize_
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize_
])

# %%
pretrained_net = torchvision.models.resnet18(pretrained=True)
pretrained_net.fc

# %%

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

# %%
""" 微调模型 """
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_dir / 'train', transform=train_augs), 
        batch_size=batch_size, shuffle=True
    ) 
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_dir / 'test', transform=test_augs), 
        batch_size=batch_size
    ) 
    
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')

    # 如果 `param_group=True`，输出层中的模型参数将使用十倍的学习率
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                        if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([
            {'params': params_1x},
            {'params': net.fc.parameters(), 'lr':learning_rate*10}
        ], lr=learning_rate, weight_decay=.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=.001)
    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

    

# %%
train_fine_tuning(finetune_net, 5e-5)

# %%
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)

# %%
