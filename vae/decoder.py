from configs.training_config import HIDDEN_DIM, LATENT_DIM
from torch import nn
from torch.nn import functional as f


class Decoder(nn.Module):

    def __init__(self, features: int, max_len: int):
        super(Decoder, self).__init__()

        self.input_linear = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.output_linear = nn.Linear(HIDDEN_DIM, features)
        self.max_len = max_len

    def forward(self, z):
        h3 = f.relu(self.input_linear(z))
        h3 = h3.unsqueeze(1).repeat(1, self.max_len, 1)  # Repeat for sequence generation.
        lstm_out, _ = self.lstm(h3)
        return self.output_linear(lstm_out)
