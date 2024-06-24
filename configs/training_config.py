"""All training configurations."""

# Padding settings.
PAD_VALUE = -1  # Masked value.


# Optimizer settings.
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
BETAS = (0.5, 0.9)


# Structure settings.
DEVICE = 'cpu'
BATCH_SIZE = 2
EPOCHS = 150
DISPLAY_FREQ = 10  # Epoch frequency of loss display.
