from configs.training_config import HIDDEN_DIM, LATENT_DIM
import torch
from torch import nn
from torch.nn import functional as f


class Encoder(nn.Module):

    def __init__(self, input_dim: int):
        super(Encoder, self).__init__()

        self.input_layer = nn.Linear(input_dim, HIDDEN_DIM)
        self.mean_layer = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.log_var_layer = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x: torch.tensor):
        hidden_output = f.relu(self.input_layer(x))
        return self.mean_layer(hidden_output), self.log_var_layer(hidden_output)
