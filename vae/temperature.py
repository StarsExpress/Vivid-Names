from vae.assembly import VAE
import torch
from torch.nn import functional as f
from sklearn.preprocessing import LabelEncoder


def decode_logits(logits: torch.Tensor, temperature: float):
    logits /= temperature
    probs = f.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def create_name(
        vae: VAE,
        latent_dim: int,
        max_len: int,
        encoder: LabelEncoder,
        temperature: float,
):

    with torch.no_grad():
        z = torch.randn(1, latent_dim)  # Sample from latent space.
        sample = vae.decode(z).view(-1, max_len, len(encoder.classes_))

        sampled_name = []
        for char_logits in sample.squeeze():
            sampled_name.append(decode_logits(char_logits, temperature).item())

        return ''.join([encoder.inverse_transform([idx])[0] for idx in sampled_name if idx != 0])
