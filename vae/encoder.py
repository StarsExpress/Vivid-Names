from configs.encoder_config import *
from configs.training_config import LATENT_DIM
import torch
from torch import nn
from torch.nn import functional as f


class Encoder(nn.Module):
    """
    Encoder of VAE.

    This class encodes input data into a latent space representation.

    It uses convolutional layers to process input data, followed by linear layers.

    Output is mean and log variance of latent space. This output is needed by reparameterization trick for VAE.

    Attributes:
        conv_layer_1 (nn.Conv1d): 1st convolutional layer for feature extraction.
        conv_layer_2 (nn.Conv1d): 2nd convolutional layer for deeper feature extraction.
        linear_layer (nn.Linear): transforms convolutional layer output to a smaller dimension.
        dropout (nn.Dropout): dropout layer for regularization.
        mean_layer (nn.Linear): linear layer to produce mean of latent space representation.
        log_var_layer (nn.Linear): linear layer to produce log variance of latent space representation.

    Args:
        timesteps (int): number of timesteps in input data.
        name_type (str): type of names being encoded, used for model parameters' selection.
    """

    def __init__(self, timesteps: int, name_type: str):
        super(Encoder, self).__init__()

        self.conv_layer_1 = nn.Conv1d(
            in_channels=1,
            out_channels=ENC_HIDDEN_DIMS[name_type]["1st"],
            kernel_size=ENC_KERNEL_SIZE,
            padding=(ENC_KERNEL_SIZE - 1) // 2,
        )

        self.conv_layer_2 = nn.Conv1d(
            in_channels=ENC_HIDDEN_DIMS[name_type]["1st"],
            out_channels=ENC_HIDDEN_DIMS[name_type]["2nd"],
            kernel_size=ENC_KERNEL_SIZE,
            padding=(ENC_KERNEL_SIZE - 1) // 2,
        )

        self.linear_layer = nn.Linear(
            ENC_HIDDEN_DIMS[name_type]["2nd"] * timesteps,
            ENC_HIDDEN_DIMS[name_type]["2nd"],
        )

        self.dropout = nn.Dropout(p=ENC_DROPOUT[name_type], inplace=True)

        self.mean_layer = nn.Linear(
            ENC_HIDDEN_DIMS[name_type]["2nd"],
            LATENT_DIM[name_type],
        )

        self.log_var_layer = nn.Linear(
            ENC_HIDDEN_DIMS[name_type]["2nd"],
            LATENT_DIM[name_type],
            )

    def forward(self, input_tensor: torch.Tensor):
        """
        Performs forward pass of encoder, encoding input data into latent space representation.

        This method processes input data through convolutional layers, followed by a linear layer to produce
        mean and log variance of latent space representation. Mean and log variance are needed in reparameterization.

        Args:
            input_tensor (torch.Tensor): shape (batch_size, timesteps).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: contains mean and log variance of latent space representation.
            Both tensors have shape (batch_size, latent_dimensions).
        """
        input_tensor = input_tensor.unsqueeze(1)  # Change shape to (batch_size, 1, timesteps) for CNN.

        hidden_output = f.leaky_relu(self.conv_layer_1(input_tensor), ENC_NEGATIVE_SLOPE)

        hidden_output = f.leaky_relu(self.conv_layer_2(hidden_output), ENC_NEGATIVE_SLOPE)

        hidden_output = hidden_output.view(hidden_output.size(0), -1)  # Flatten for linear layer.
        hidden_output = self.linear_layer(hidden_output)
        self.dropout(hidden_output)
        return self.mean_layer(hidden_output), self.log_var_layer(hidden_output)
