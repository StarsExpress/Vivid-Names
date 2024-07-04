from configs.training_config import LATENT_DIM
from vae.assembly import VAE
import torch
from torch.nn import functional as f
from sklearn.preprocessing import LabelEncoder


def decode_logits(logits: torch.Tensor, temperature: float):
    """
    Decode logits into probabilities and samples from them.

    Args:
        logits (torch.Tensor): logits to be decoded, shape (features).
        temperature (float): temperature to expand/shrink logits before entering softmax.

    Returns:
        int: sampled index.
    """
    logits /= temperature
    probs = f.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()  # Tensor to integer.


def create_name(vae: VAE, encoder: LabelEncoder, temperature: float, names_type: str):
    """
    Creates a name using a trained VAE model and a label encoder.

    This function takes a random sample from latent space, decodes it into a sequence of logits
    representing character probabilities. Characters sampling is based on probabilities adjusted by temperature.

    Temperature controls randomness of sampling process, with lower values leading to less random outputs.

    Finally, a label encoder converts sampled character indices back into character strings.

    Args:
        vae (VAE): VAE model used for creating names.
        encoder (LabelEncoder): label encoder used for indices conversion back into characters.
        temperature (float): used for controlling randomness of character sampling.
        names_type (str): type of names to create, which determines latent space to sample from.

    Returns:
        str: created name.
    """
    with torch.no_grad():
        sample = torch.randn(1, LATENT_DIM[names_type])  # Sample from latent space.
        # Detach gradients. Squeeze away any singleton dimension.
        decoded_sample = vae.decode(sample).detach().squeeze()  # Shape: (1, max_len, features).

        char_indices = []
        for char_logits in decoded_sample:
            char_indices.append(decode_logits(char_logits, temperature))

        creation = ""
        for idx in char_indices:
            if idx != 0:  # Skip padded char. [0] gets decoded char.
                creation += encoder.inverse_transform([idx])[0]
        return creation
