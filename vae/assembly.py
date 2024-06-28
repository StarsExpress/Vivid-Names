from vae.encoder import Encoder
from vae.decoder import Decoder
import torch
from torch import nn


class VAE(nn.Module):

    def __init__(self, input_dim: int, max_len: int, features: int, names_type: str):
        super(VAE, self).__init__()
        self.features = features

        self.encoder = Encoder(input_dim, names_type)
        self.decoder = Decoder(features, max_len, names_type)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    @staticmethod
    def reparameterize(latent_mean, latent_log_var):
        std = torch.exp(0.5 * latent_log_var)
        epsilon = torch.randn_like(std)
        return latent_mean + epsilon * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        latent_mean, latent_log_var = self.encode(x.view(-1, x.size(1)))
        z = self.reparameterize(latent_mean, latent_log_var)
        return self.decode(z), latent_mean, latent_log_var
