from utils.files_helper import read_unique_names
import numpy as np


surnames = read_unique_names('surnames')


def select_surnames(size: int):
    """
    Select a number of surnames from surnames list via uniform probability.

    Args:
        size (int): number of surnames to select.

    Returns:
        list: list of selected surnames.
    """
    return np.random.choice(surnames, size=size)  # Return entire list for remix preference.
