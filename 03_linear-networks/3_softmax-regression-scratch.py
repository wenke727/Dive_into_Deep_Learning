#%%
from matplotlib import animation
from matplotlib.pyplot import axes, ylim
from sklearn import metrics
import torch
from IPython import display
from d2l import torch as d2l

# %%
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# %%
# initial params
num_input = 784
num_output = 10

W = torch.normal(0, .01, size=(num_input, num_output), requires_grad=True)
b = torch.zeros(num_output, requires_grad=True)

# %%
# 定义softmax操作
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    
    return X_exp / partition

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)

# %%
# 定义模型

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

#%%
# loss
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]

cross_entropy(y_hat, y)

# %%
# 分类精度

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)

# %%
class Accumulator:
    def __init__(self, n):
        self.data = [.0] * n
    
    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]
        
    def reset(self):
        self.data = [.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
        
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())

    return metric[0] / metric[1]

evaluate_accuracy(net, test_iter)
    
# %%
lr = .1

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols,figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        
        if not hasattr(x, "__len__"):
            x = [x] * n
            
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        
        for  i, (a, b) in enumerate(zip(x,y)):
            if a is None or b is None:
                continue
            self.X[i].append(a)
            self.Y[i].append(b)
        
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    if isinstance(net,torch.nn.Module):
        net.train()
    
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            # FIXME
            metric.add(float(l)*len(y), accuracy(y_hat, y), y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel()) 

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[.3, .9], legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
        
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
        
    
# %%
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)

# %%
