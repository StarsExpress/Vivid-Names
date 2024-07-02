from configs.decoder_config import DEC_DROPOUT
from configs.training_config import LATENT_DIM
import torch
from torch import nn
from torch.nn import functional as f


class Decoder(nn.Module):
    """
    Decoder part of VAE.

    Responsible for decoding reparameterized latent space back into original data space.

    Attributes:
        max_len (int): max length for a name. All names will be padded to this length.
        input_layer (nn.Linear): input layer of decoder.
        lstm_layer (nn.LSTM): LSTM layer of decoder.

    Methods:
        forward(z: torch.Tensor): performs a forward pass through decoder.
    """

    def __init__(self, features: int, max_len: int, names_type: str):
        """
        Initialize decoder with given features, max length, and names type.

        Args:
            features (int): number of features in input data.
            max_len (int): max length for a name. All names will be padded to this length.
            names_type (str): type of names to be created. Can be: 'surnames', 'female_forenames', 'male_forenames'.
        """
        super(Decoder, self).__init__()

        self.max_len, self.names_type = max_len, names_type

        self.input_layer = nn.Linear(LATENT_DIM[names_type], features)

        self.batch_norm = nn.BatchNorm1d(features)
        self.dropout = nn.Dropout(p=DEC_DROPOUT[names_type])

        self.lstm_layer = nn.LSTM(
            features,
            features,
            batch_first=True,
            dropout=DEC_DROPOUT[self.names_type],
        )
        self.output_layer = nn.Linear(features, features)

    def forward(self, z: torch.tensor):
        """
        Performs a forward pass through decoder.

        Args:
            z (torch.tensor): reparameterized latent space.

        Returns:
            torch.Tensor: decoded data.
        """
        hidden_output = self.input_layer(z)
        hidden_output = f.gelu(self.batch_norm(hidden_output))
        hidden_output = self.dropout(hidden_output)

        # Repeat for sequence generation.
        hidden_output = hidden_output.unsqueeze(1).repeat(1, self.max_len, 1)

        lstm_output, _ = self.lstm_layer(hidden_output)
        return self.output_layer(lstm_output)
