import torch
from torch.nn import functional as f


def compute_loss(
        reconstructed_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
):
    bce = f.cross_entropy(
        reconstructed_x.view(-1, reconstructed_x.size(-1)),
        x.view(-1),
        reduction='sum'
    )  # Reconstruction loss.
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # Regularization loss.
    return bce + kld
