from configs.training_config import HIDDEN_DIM, LATENT_DIM
from torch import nn
from torch.nn import functional as f


class Encoder(nn.Module):

    def __init__(self, input_dim: int):
        super(Encoder, self).__init__()

        self.input_linear = nn.Linear(input_dim, HIDDEN_DIM)
        self.mu_linear = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.output_linear = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x):
        h1 = f.relu(self.input_linear(x))
        return self.mu_linear(h1), self.output_linear(h1)
