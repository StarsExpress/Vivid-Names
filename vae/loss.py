from configs.training_config import VAE_BETA
import torch
from torch.nn import functional as f


def compute_loss(
    reconstructed_input: torch.Tensor,
    original_input: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_log_var: torch.Tensor,
    names_type: str,
):
    """
    Compute loss for Beta VAE. Loss is composed of two parts:

    1. Reconstruction loss: Binary Cross Entropy between reconstructed input and original input.

    2. Regularization loss: Kullback-Leibler Divergence between learned latent and prior distribution.
       Regularization loss will be weighted by Beta parameter.

    Args:
        reconstructed_input (torch.Tensor): reconstructed input, shape (batch_size, timesteps, features).
        original_input (torch.Tensor): original input, shape (batch_size, timesteps).
        latent_mean (torch.Tensor): mean of latent distribution, shape (batch_size, latent_dimensions).
        latent_log_var (torch.Tensor): log variance of latent distribution, shape (batch_size, latent_dimensions).
        names_type (str): type of names to be created. Can be: 'surnames', 'female_forenames', 'male_forenames'.

    Returns:
        torch.Tensor: computed loss.
    """
    cce = f.cross_entropy(
        reconstructed_input.reshape(-1, reconstructed_input.size(-1)),
        original_input.reshape(-1),
        reduction="sum",
    )
    kld = -0.5 * torch.sum(
        1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp()
    )
    return cce + VAE_BETA[names_type] * kld
