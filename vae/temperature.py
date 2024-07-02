from configs.training_config import LATENT_DIM
from vae.assembly import VAE
import torch
from torch.nn import functional as f
from sklearn.preprocessing import LabelEncoder


def decode_logits(logits: torch.Tensor, temperature: float):
    """
    Decode logits into probabilities and samples from them.

    Args:
        logits (torch.Tensor): logits to be decoded.
        temperature (float): temperature to expand/shrink logits before entering softmax.

    Returns:
        int: sampled index.
    """
    logits /= temperature
    probs = f.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()  # Tensor to integer.


def interpolate(mu1, mu2):
    ratios = torch.linspace(0, 1, 10)
    return [(1 - r) * mu1 + r * mu2 for r in ratios]


def create_name(vae: VAE, encoder: LabelEncoder, temperature: float, names_type: str):
    """
    Create a name using VAE and LabelEncoder.

    Args:
        vae (VAE): VAE for names creation.
        encoder (LabelEncoder): LabelEncoder to use for decoding created name.
        temperature (float): temperature to use for softmax.
        names_type (str): type of names to be created. Can be: 'surnames', 'female_forenames', 'male_forenames'.

    Returns:
        str: created name.
    """
    with torch.no_grad():
        sample = torch.randn(1, LATENT_DIM[names_type])  # Sample from latent space.
        # Squeeze away any singleton dimension.
        decoded_sample = vae.decode(sample).squeeze()  # Shape: (1, max_len, features).

        char_indices = []
        for char_logits in decoded_sample:
            char_indices.append(decode_logits(char_logits, temperature))

        creation = ""
        for idx in char_indices:
            if idx != 0:  # Skip padded char. [0] gets decoded char.
                creation += encoder.inverse_transform([idx])[0]
        return creation
