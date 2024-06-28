"""All training configurations."""

# Latent space settings.
HIDDEN_DIM = 100
LATENT_DIM = {
    "surnames": 20,
    "female_forenames": 80,  # Forenames are more entangled.
    "male_forenames": 80,
}


# Regularization loss settings.
VAE_BETA = {
    "surnames": 1.5,
    "female_forenames": 1.5,
    "male_forenames": 0.8,
}


# Optimizer settings.
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
BETAS = (0.5, 0.9)


# Epoch settings.
DEVICE = "cpu"
BATCH_SIZE = 2
EPOCHS = 100
DISPLAY_FREQ = 10  # Epoch frequency of loss display.
