from configs.training_config import VAE_BETA
import torch
from torch.nn import functional as f


def compute_loss(
    reconstructed_x: torch.Tensor,
    x: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_log_var: torch.Tensor,
    names_type: str,
):
    """
    Compute loss for VAE. Loss is composed of two parts:

    1. Reconstruction loss: Binary Cross Entropy between reconstructed input and original input.

    2. Regularization loss: Kullback-Leibler Divergence between learned latent distribution and prior distribution.

    Args:
        reconstructed_x (torch.Tensor): reconstructed input.
        x (torch.Tensor): original input.
        latent_mean (torch.Tensor): mean of the learned latent distribution.
        latent_log_var (torch.Tensor): log variance of the learned latent distribution.
        names_type (str): type of names to be created. Can be: 'surnames', 'female_forenames', 'male_forenames'.

    Returns:
        torch.Tensor: computed loss.
    """
    cce = f.cross_entropy(
        reconstructed_x.view(-1, reconstructed_x.size(-1)),
        x.view(-1),
        reduction="sum",
    )
    kld = -0.5 * torch.sum(
        1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp()
    )
    return cce + VAE_BETA[names_type] * kld
