import torch
import pyro


def auto_seed(seed):
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2 ** 30)
        pyro.set_rng_seed(seed)
    return seed


def detach_gradients(module, val):
    for p in module.parameters():
        p.requires_grad = not val
