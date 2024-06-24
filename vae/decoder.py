from torch import nn
from torch.nn import functional as f


class Decoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, vocab_size, max_len):
        super(Decoder, self).__init__()

        self.input_linear = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len

    def forward(self, z):
        h3 = f.relu(self.input_linear(z))
        h3 = h3.unsqueeze(1).repeat(1, self.max_len, 1)  # Repeat for sequence generation.
        lstm_out, _ = self.lstm(h3)
        return self.output_linear(lstm_out)
