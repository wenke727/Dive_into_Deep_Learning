#%%
import random
import torch
from d2l import torch as d2l

# %%

def sythetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, .01, y.shape)
    
    return X, y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = sythetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

# %%
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].numpy(), labels.numpy(), 1);

# %%

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_idxs = torch.tensor(
            indices[i: min(i+batch_size, num_examples)]
        )
        
        yield features[batch_idxs], labels[batch_idxs]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# %%

# initial params
w = torch.normal(0, 0, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linereg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat-y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# %%
# training
lr = .03
num_epoches = 3
net = linereg
loss = squared_loss

for epoch in range(num_epoches):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# %%
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

# %%
