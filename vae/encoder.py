from configs.training_config import HIDDEN_DIM, LATENT_DIM
import torch
from torch import nn
from torch.nn import functional as f


class Encoder(nn.Module):
    """
    Encoder part of VAE.

    Responsible for encoding input data into latent space.

    Attributes:
        input_layer (nn.Linear): input layer of encoder.
        mean_layer (nn.Linear): layer that calculates mean of latent space.
        log_var_layer (nn.Linear): layer that calculates log variance of latent space.

    Methods:
        forward(x: torch.Tensor): performs a forward pass through encoder.
    """

    def __init__(self, input_dim: int, names_type: str):
        """
        Initialize encoder with given input dimension and names type.

        Args:
            input_dim (int): dimension of input data.
            names_type (str): type of names to be created. Can be: 'surnames', 'female_forenames', 'male_forenames'.
        """
        super(Encoder, self).__init__()

        self.input_layer = nn.Linear(input_dim, HIDDEN_DIM)
        self.mean_layer = nn.Linear(HIDDEN_DIM, LATENT_DIM[names_type])
        self.log_var_layer = nn.Linear(HIDDEN_DIM, LATENT_DIM[names_type])

    def forward(self, x: torch.tensor):
        """
        Performs a forward pass through encoder.

        Args:
            x (torch.tensor): input data.

        Returns:
            tuple: mean and log variance of latent space.
        """
        hidden_output = f.selu(self.input_layer(x))
        return self.mean_layer(hidden_output), self.log_var_layer(hidden_output)
