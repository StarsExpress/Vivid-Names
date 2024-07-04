"""All encoder configurations."""

# Neural nets settings.
ENC_HIDDEN_DIMS = {
    "surnames": {"1st": 45, "2nd": 35},
    "female_forenames": {"1st": 50, "2nd": 40},
    "male_forenames": {"1st": 50, "2nd": 40},
}

ENC_KERNEL_SIZE = 3

ENC_DROPOUT = {
    "surnames": 0.15,
    "female_forenames": 0.2,
    "male_forenames": 0.1,
}

ENC_NEGATIVE_SLOPE = 0.15
