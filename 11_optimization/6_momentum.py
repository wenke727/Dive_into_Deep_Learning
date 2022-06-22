#%%
import torch
from d2l import torch as d2l

# %%
def init_momentum_states(feature_dim):
    w = torch.zeros((feature_dim, 1))
    b = torch.ones(1)
    
    return (w, b)


def sgd_momentum(params, states, hyperparams):
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = hyperparams['momentum'] * s + p.grad
            p[:] -= hyperparams['lr'] * s
        
        p.grad.data.zero_()
        

def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(trainer_fn=sgd_momentum, 
                   states=init_momentum_states(feature_dim), 
                   hyperparams={'lr': lr, 'momentum': momentum}, 
                   data_iter=data_iter,
                   feature_dim=feature_dim, 
                   num_epochs=num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

train_momentum(lr=0.02, momentum=0.5)
train_momentum(lr=0.005, momentum=0.9)

# %%
""" 简洁实现 """

trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)

# %%
