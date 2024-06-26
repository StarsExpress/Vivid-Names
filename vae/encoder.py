from configs.training_config import HIDDEN_DIM, LATENT_DIM
import torch
from torch import nn
from torch.nn import functional as f


class Encoder(nn.Module):

    def __init__(self, input_dim: int):
        super(Encoder, self).__init__()

        self.input_linear = nn.Linear(input_dim, HIDDEN_DIM)
        self.mu_linear = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.output_linear = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x: torch.tensor):
        hidden_output = f.relu(self.input_linear(x))
        return self.mu_linear(hidden_output), self.output_linear(hidden_output)
