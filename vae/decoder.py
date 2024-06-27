from configs.training_config import HIDDEN_DIM, LATENT_DIM
import torch
from torch import nn
from torch.nn import functional as f


class Decoder(nn.Module):

    def __init__(self, features: int, max_len: int):
        super(Decoder, self).__init__()

        self.max_len = max_len

        self.input_layer = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.lstm_layer = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.output_layer = nn.Linear(HIDDEN_DIM, features)

    def forward(self, z: torch.tensor):
        hidden_output = f.gelu(self.input_layer(z))
        # Repeat for sequence generation.
        hidden_output = hidden_output.unsqueeze(1).repeat(1, self.max_len, 1)
        lstm_output, _ = self.lstm_layer(hidden_output)
        return self.output_layer(lstm_output)
