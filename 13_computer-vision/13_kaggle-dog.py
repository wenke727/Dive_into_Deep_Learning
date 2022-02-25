#%%
%matplotlib inline
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

# %%
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip', '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# 如果你使用Kaggle比赛的完整数据集，请将下面的变量更改为False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')

# %%
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


""" img aug """
tranform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(.08, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.4),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225] )
])

tranform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225] )
])


train_ds, train_valid_ds = [
    ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=tranform_train) 
        for folder in ['train', 'train_valid']
]

valid_ds, test_ds = [
    ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=tranform_test) 
        for folder in ['valid', 'test']
]


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

train_iter, train_valid_iter = [
    DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True
    ) for dataset in (train_ds, train_valid_ds)
]

valid_iter = DataLoader(valid_ds, batch_size, shuffle=True, drop_last=True)
test_iter = DataLoader(test_ds, batch_size, shuffle=True, drop_last=False)

loss = nn.CrossEntropyLoss(reduction='none')

# %%

def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # finetune_net.features = torchvision.models.resnet50(pretrained=True)
    
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    
    finetune_net = finetune_net.to(devices[0])
    
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    
    return finetune_net


def evaluate_loss(data_iter, net, devices):
    l_sum, n = .0, 0
    for X, y in data_iter:
        X, y = X.to(devices[0]), y.to(devices[0])

        outputs = net(X)
        l = loss(outputs, y)
        l_sum += l.sum()
        n += y.numel()
    
    return (l_sum /n).to('cpu') 



# %%
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    # 只训练小型自定义输出网络
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            output = net(features)

            l = loss(output, labels).sum()
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
    
# %%
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# %%
net  = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)


# %%
preds = []
for data, label in test_iter:
    output = F.softmax(net(data.to(devices[0])), dim=0)
    preds.extend(output.cpu().detach().numpy())
# ids = sorted()
preds

# %%
