from configs.training_config import LATENT_DIM
from vae.assembly import VAE
import torch
from torch.nn import functional as f
from sklearn.preprocessing import LabelEncoder


def decode_logits(logits: torch.Tensor, temperature: float):
    logits /= temperature
    probs = f.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()  # Tensor to integer.


def create_name(vae: VAE, encoder: LabelEncoder, temperature: float, names_type: str):
    with torch.no_grad():
        sample = torch.randn(1, LATENT_DIM[names_type])  # Sample from latent space.
        # Squeeze away any singleton dimension.
        decoded_sample = vae.decode(sample).squeeze()  # Shape: (1, max_len, features).

        char_indices = []
        for char_logits in decoded_sample:
            char_indices.append(decode_logits(char_logits, temperature))

        created_chars = []
        for idx in char_indices:
            if idx != 0:  # Skip padded char. [0] gets decoded char.
                created_chars.append(encoder.inverse_transform([idx])[0])
        return ''.join(created_chars)
