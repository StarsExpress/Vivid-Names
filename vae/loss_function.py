import torch
from torch.nn import functional as f


def loss_function(
        reconstruction_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
):
    bce = f.cross_entropy(
        reconstruction_x.view(-1, reconstruction_x.size(-1)),
        x.view(-1),
        reduction='sum'
    )
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld
