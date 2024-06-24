from vae.encoder import Encoder
from vae.decoder import Decoder
import torch
from torch import nn


class Assembly(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, max_len, vocab_size):
        super(Assembly, self).__init__()
        self.vocab_size = vocab_size

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(hidden_dim, latent_dim, vocab_size, max_len)

    def encode(self, x):
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, x.size(1)))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
