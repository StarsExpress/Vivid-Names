from configs.encoder_config import ENC_HIDDEN_DIM, ENC_DROPOUT
from configs.training_config import LATENT_DIM
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
        self.names_type = names_type

        self.input_layer = nn.Linear(input_dim, ENC_HIDDEN_DIM[names_type])
        self.batch_norm = nn.BatchNorm1d(ENC_HIDDEN_DIM[names_type])
        self.dropout = nn.Dropout(p=ENC_DROPOUT[names_type])

        self.mean_layer = nn.Linear(ENC_HIDDEN_DIM[names_type], LATENT_DIM[names_type])
        self.log_var_layer = nn.Linear(ENC_HIDDEN_DIM[names_type], LATENT_DIM[names_type])

    def forward(self, x: torch.tensor):
        """
        Performs a forward pass through encoder.

        Args:
            x (torch.tensor): input data.

        Returns:
            tuple: mean and log variance of latent space.
        """
        hidden_output = self.batch_norm(self.input_layer(x))
        hidden_output = self.dropout(f.relu6(hidden_output))
        return self.mean_layer(hidden_output), self.log_var_layer(hidden_output)
