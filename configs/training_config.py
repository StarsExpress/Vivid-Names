"""All training configurations."""

# Latent space settings.
LATENT_DIM = {
    "surnames": 25,
    "female_forenames": 30,
    "male_forenames": 30,
}


# Regularization loss settings.
VAE_BETA = {
    "surnames": 2.0,
    "female_forenames": 2.5,
    "male_forenames": 2.0,
}


# Optimizer settings.
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
BETAS = (0.5, 0.9)


# Epoch settings.
BATCH_SIZE = 10
EPOCHS = 150
DISPLAY_FREQ = 10  # Epoch frequency of loss display.
