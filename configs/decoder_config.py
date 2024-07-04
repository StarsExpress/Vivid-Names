"""All decoder configurations."""

# Neural nets settings.
DEC_HIDDEN_DIMS = {
    "surnames": {"1st": 35, "2nd": 45},
    "female_forenames": {"1st": 40, "2nd": 50},
    "male_forenames": {"1st": 40, "2nd": 50},
}

DEC_KERNEL_SIZE = 3

DEC_DROPOUT = {
    "surnames": 0.1,
    "female_forenames": 0.2,
    "male_forenames": 0.15,
}

DEC_NEGATIVE_SLOPE = 0.2
