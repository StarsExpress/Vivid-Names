"""All training configurations."""

# Optimizer settings.
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 0.001
BETAS = (0.5, 0.9)


# Structure settings.
DEVICE = 'cpu'
HIDDEN_DIM = 256
LATENT_DIM = 20
BATCH_SIZE = 2
EPOCHS = 200
DISPLAY_FREQ = 10  # Epoch frequency of loss display.
