import torch
from torch import nn
from torch.nn import functional as f
from configs.decoder_config import *
from configs.training_config import LATENT_DIM


class Decoder(nn.Module):
    """
    Decoder of VAE.

    This class decodes latent space representation back into original input space.

    It uses a series of deconvolutional layers to reconstruct data from latent dimensions.

    Attributes:
        name_type (str): type of names data being decoded, used for model parameters' selection.
        timesteps (int): number of timesteps in input data, used for reshaping output.
        linear_layer (nn.Linear): transforms latent variables to a shape suitable for deconvolution.
        deconv_layer_1 (nn.ConvTranspose1d): 1st deconvolutional layer for reconstruction.
        deconv_layer_2 (nn.ConvTranspose1d): 2nd deconvolutional layer for reconstruction.
        dropout (nn.Dropout): dropout layer for regularization.

    Methods:
        forward(z: torch.Tensor): forward pass of decoder, reconstructing input data from latent variables.

    Args:
        timesteps (int): number of timesteps in input data.
        features (int): number of features in input data.
        name_type (str): type of names data being decoded.
    """

    def __init__(self, timesteps: int, features: int, name_type: str):
        """
        Initializes decoder module with necessary layers and parameters.

        This method sets up linear layer, deconvolutional layers and a dropout layer.

        Args:
            timesteps (int): used for reshaping output of deconvolutional layers.
            features (int): corresponds to output size of final deconvolutional layer.
            name_type (str): used to select specific model parameters from configu.
        """

        super(Decoder, self).__init__()
        self.name_type, self.timesteps = name_type, timesteps

        self.linear_layer = nn.Linear(
            LATENT_DIM[name_type], DEC_HIDDEN_DIMS[name_type]["1st"] * timesteps
        )

        self.deconv_layer_1 = nn.ConvTranspose1d(
            DEC_HIDDEN_DIMS[name_type]["1st"],
            DEC_HIDDEN_DIMS[name_type]["2nd"],
            kernel_size=DEC_KERNEL_SIZE,
            padding=(DEC_KERNEL_SIZE - 1) // 2,
        )

        self.deconv_layer_2 = nn.ConvTranspose1d(
            DEC_HIDDEN_DIMS[name_type]["2nd"],
            features,
            kernel_size=DEC_KERNEL_SIZE,
            padding=(DEC_KERNEL_SIZE - 1) // 2,
        )

        self.dropout = nn.Dropout(p=DEC_DROPOUT[name_type], inplace=True)

    def forward(self, latent_variables: torch.Tensor):
        """
        Performs forward pass of decoder, reconstructing input data from latent variables.

        This method first upscales latent variables by a linear layer.

        Then applies two deconvolutional layers with leaky ReLU and dropout.
        Activations can't be done inplace due to gradients, but dropout can.

        Output is permuted to match expected shape of (batch_size, timesteps, features).

        Args:
            latent_variables (torch.Tensor): shape (batch_size, latent_dimensions).

        Returns:
            torch.Tensor: reconstructed data. Shape (batch_size, timesteps, features).
        """
        hidden_output = self.linear_layer(latent_variables)

        hidden_output = hidden_output.view(
            latent_variables.size(0),
            DEC_HIDDEN_DIMS[self.name_type]["1st"],
            self.timesteps,
        )
        hidden_output = f.leaky_relu(self.deconv_layer_1(hidden_output), DEC_NEGATIVE_SLOPE)

        hidden_output = f.leaky_relu(self.deconv_layer_2(hidden_output), DEC_NEGATIVE_SLOPE)
        self.dropout(hidden_output)
        return hidden_output.permute(0, 2, 1)  # Permute to shape: (batch_size, timesteps, features).
