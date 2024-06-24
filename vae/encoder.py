from torch import nn
from torch.nn import functional as f


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.mu_linear = nn.Linear(hidden_dim, latent_dim)
        self.output_linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h1 = f.relu(self.input_linear(x))
        return self.mu_linear(h1), self.output_linear(h1)
