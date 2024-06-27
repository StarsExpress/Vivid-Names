"""All training configurations."""

# Latent space settings.
HIDDEN_DIM = 100
LATENT_DIM = 20
BETA = 1.5  # Regularization loss weight.


# Optimizer settings.
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
BETAS = (0.5, 0.9)


# Structure settings.
DEVICE = 'cpu'
BATCH_SIZE = 2
EPOCHS = 200
DISPLAY_FREQ = 10  # Epoch frequency of loss display.
