from configs.training_config import BETA
import torch
from torch.nn import functional as f


def compute_loss(
        reconstructed_x: torch.Tensor,
        x: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_log_var: torch.Tensor,
):
    bce = f.cross_entropy(
        reconstructed_x.view(-1, reconstructed_x.size(-1)),
        x.view(-1),
        reduction='sum',
    )

    kld = 0.5 * torch.sum(1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp())
    return bce - BETA * kld
