import torch
from torch.nn import functional as f


def decode_logits(logits, temperature):
    logits = logits / temperature
    probs = f.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def create_name(vae, latent_dim, max_len, encoder, temperature):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim)  # Sample from the latent space
        sample = vae.decode(z).view(-1, max_len, len(encoder.classes_))
        sampled_name = [decode_logits(char_logits, temperature).item() for char_logits in sample.squeeze()]
        return ''.join([encoder.inverse_transform([idx])[0] for idx in sampled_name if idx != 0])
