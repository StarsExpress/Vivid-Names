"""All training configurations."""

# Latent space settings.
LATENT_DIM = {
    "surnames": 30,
    "female_forenames": 30,
    "male_forenames": 35,
}


# Regularization loss settings.
VAE_BETA = {
    "surnames": 1.0,
    "female_forenames": 1.5,
    "male_forenames": 1.3,
}


# Optimizer settings.
LEARNING_RATE = 0.0015
WEIGHT_DECAY = 0.001
BETAS = (0.5, 0.9)


# Epoch settings.
BATCH_SIZE = 10
DISPLAY_FREQ = 50  # Epoch frequency of loss display.
