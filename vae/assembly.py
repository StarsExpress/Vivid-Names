from vae.encoder import Encoder
from vae.decoder import Decoder
import torch
from torch import nn


class VAE(nn.Module):
    """
    Variational Autoencoder class.

    This class provides methods for encoding input data into latent space,
    reparameterizing latent space to enable gradient descent,
    and decoding from latent space to input space.

    Attributes:
        features (int): number of features in input data.
        encoder (Encoder): encoder of VAE.
        decoder (Decoder): decoder of VAE.

    Methods:
        __init__(timesteps: int, features: int, names_type: str): initializes VAE with encoder and decoder.
        encode(input_tensor: torch.Tensor): encodes input data into latent space.
        reparameterize(latent_mean: torch.Tensor, latent_log_var: torch.Tensor): reparameterizes latent space.
        decode(latent_variables: torch.Tensor): decodes latent variables back to input space.
        forward(x: torch.Tensor): forward pass of VAE.

    Args:
        timesteps (int): number of timesteps in input data.
        features (int): number of features in input data.
        name_type (str): type of names being processed.
    """

    def __init__(self, timesteps: int, features: int, name_type: str):
        """
        Initializes VAE with encoder and decoder.

        Args:
            timesteps (int): length of longest name in input data.
            features (int): number of unique characters appearing in input data.
            name_type (str): type of names being processed.
        """
        super(VAE, self).__init__()
        self.features = features

        self.encoder = Encoder(timesteps, name_type)
        self.decoder = Decoder(timesteps, features, name_type)

    def encode(self, input_tensor: torch.Tensor):
        """
        Encodes input data into latent space during training.

        Args:
            input_tensor (torch.Tensor): shape (batch_size, timesteps).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: contains latent mean and log variance tensors.

            Both tensors have shape (batch_size, latent_dimensions).
        """
        return self.encoder(input_tensor)

    @staticmethod
    def reparameterize(latent_mean, latent_log_var):
        """
        Reparameterizes latent space to enable gradient descent by adding stochastic component to latent encoding.

        Args:
            latent_mean (torch.Tensor): mean of latent variables. Shape (batch_size, latent_dimensions).
            latent_log_var (torch.Tensor): log variance of latent variables. Shape (batch_size, latent_dimensions).

        Returns:
            torch.Tensor: reparameterized latent variables. Shape (batch_size, latent_dimensions).
        """
        std = torch.exp(0.5 * latent_log_var)
        epsilon = torch.randn_like(std)
        return latent_mean + epsilon * std

    def decode(self, latent_variables: torch.Tensor):
        """
        Decodes given latent variables back to data space for names creation.

        Args:
            latent_variables (torch.Tensor): shape (batch_size, latent_dimensions).

        Returns:
            torch.Tensor: decoded output of shape (batch_size, output_dimensions).

            output_dimensions depends on decoder config and may not necessarily match input dimensions.
        """
        return self.decoder(latent_variables)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of VAE during training and creation.

        Args:
            x (torch.Tensor): shape (batch_size, timesteps).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: contains decoded output, latent mean and log variance.
        """
        latent_mean, latent_log_var = self.encode(x.view(-1, x.size(1)))
        z = self.reparameterize(latent_mean, latent_log_var)
        return self.decode(z), latent_mean, latent_log_var
