from configs.training_config import HIDDEN_DIM, LATENT_DIM
import torch
from torch import nn
from torch.nn import functional as f


class Decoder(nn.Module):

    def __init__(self, features: int, max_len: int):
        super(Decoder, self).__init__()

        self.input_linear = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.output_linear = nn.Linear(HIDDEN_DIM, features)
        self.max_len = max_len

    def forward(self, z: torch.tensor):
        hidden_output = f.relu(self.input_linear(z))
        # Repeat for sequence generation.
        hidden_output = hidden_output.unsqueeze(1).repeat(1, self.max_len, 1)
        lstm_out, _ = self.lstm(hidden_output)
        return self.output_linear(lstm_out)
