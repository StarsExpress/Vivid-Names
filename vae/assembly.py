from vae.encoder import Encoder
from vae.decoder import Decoder
import torch
from torch import nn


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for names creation.

    Attributes:
        features (int): number of features in input data.
        encoder (Encoder): encoder part of VAE.
        decoder (Decoder): decoder part of VAE.

    Methods:
        encode(x: torch.Tensor): encodes input data into latent space.
        reparameterize(latent_mean, latent_log_var): reparameterizes latent space.
        decode(z: torch.Tensor): decodes the reparameterized latent space back into original data space.
        forward(x: torch.Tensor): performs a forward pass through VAE.
    """

    def __init__(self, input_dim: int, max_len: int, features: int, names_type: str):
        super(VAE, self).__init__()
        self.features = features

        self.encoder = Encoder(input_dim, names_type)
        self.decoder = Decoder(features, max_len, names_type)

    def encode(self, x: torch.Tensor):
        """
        Encodes input data into latent space.

        Args:
            x (torch.Tensor): input data.

        Returns:
            tuple: mean and log variance of latent space.
        """
        return self.encoder(x)

    @staticmethod
    def reparameterize(latent_mean, latent_log_var):
        """
        Reparameterizes latent space.

        Args:
            latent_mean: mean of latent space.
            latent_log_var: log variance of latent space.

        Returns:
            torch.Tensor: reparameterized latent space.
        """
        std = torch.exp(0.5 * latent_log_var)
        epsilon = torch.randn_like(std)
        return latent_mean + epsilon * std

    def decode(self, z: torch.Tensor):
        """
        Decodes reparameterized latent space back into original data space.

        Args:
            z (torch.Tensor): reparameterized latent space.

        Returns:
            torch.Tensor: decoded data.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass through VAE.

        Args:
            x (torch.Tensor): input data.

        Returns:
            tuple: decoded data, mean of latent space, and log variance of latent space.
        """
        latent_mean, latent_log_var = self.encode(x.view(-1, x.size(1)))
        z = self.reparameterize(latent_mean, latent_log_var)
        return self.decode(z), latent_mean, latent_log_var
