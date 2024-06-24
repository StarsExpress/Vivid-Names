import torch
from torch import nn
from torch.nn import functional as f


def loss_function(recon_x, x, mu, logvar):
    bce = f.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld
