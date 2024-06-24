from utils.files_helper import read_unique_names
import numpy as np


surnames = read_unique_names('surnames').tolist()


def adjust_creation(creation: str):
    """
    Adjust created name by capitalizing name. If creation starts with Mc, capitalize 3rd letter.

    Args:
        creation (str): created name.

    Returns:
        str: adjusted name.
    """
    creation = creation.capitalize()
    if creation[:2] == 'Mc':
        creation = creation[:2] + creation[2:].capitalize()
    return creation


def select_surnames(size: int):
    """
    Select a number of surnames from surnames list via uniform probability.

    Args:
        size (int): number of surnames to select.

    Returns:
        list: list of selected surnames.
    """
    return np.random.choice(surnames, size=size)  # Return entire list for remix preference.
