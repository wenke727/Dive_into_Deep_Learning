#%%
import torch
from d2l import torch as d2l

# %%
def init_adagrad_states(feature_dim):
    w = torch.zeros((feature_dim, 1))
    b = torch.ones(1)
    
    return w, b

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
    p.grad.data.zero_()
    

def train_adagrad(lr, num_epochs=2):
    d2l.train_ch11(trainer_fn=adagrad, 
                   states=init_adagrad_states(feature_dim), 
                   hyperparams={'lr': lr}, 
                   data_iter=data_iter,
                   feature_dim=feature_dim, 
                   num_epochs=num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

train_adagrad(.1)

# %%
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
# %%
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)

# %%
