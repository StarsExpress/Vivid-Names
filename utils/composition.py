from creators.surnames_creator import SurnamesCreator
from creators.forenames_creator import ForenamesCreator
from utils.files_helper import read_unique_names
import numpy as np
import time


surnames_list = read_unique_names("surnames")
surname_creator = SurnamesCreator()
forename_creators_dict = {
    "male": ForenamesCreator("male"), "female": ForenamesCreator("female"),
}


def select_surnames(size: int):
    """
    For remix usage, select surnames via uniform probability.

    Args:
        size (int): number of surnames to select.

    Returns:
        list: list of selected surnames.
    """
    return np.random.choice(surnames_list, size=size)


def make_creations(num_names: int, temperature: float, gender: str = "female", target: str = "remix"):
    """
    Create a specified number of names based on provided parameters.

    Also measure total and average time taken to create names.

    Args:
        num_names (int): number of names to create.
        temperature (float): creativity level. Higher means more creative/randomness.
        gender (str, optional): gender for creation. Defaults to 'female'.
        target (str, optional): target for creation.
        Can be 'remix', 'just_surname', 'just_forename', or 'full_name'. Defaults to 'remix'.

    Returns:
        tuple: list of created names, total time and average time to create names.
    """

    start = time.time()
    surnames, forenames = [], []

    if target != "just_forename":  # If not just forename, surnames must be needed.
        if target == "remix":  # Select from existing surnames.
            surnames = select_surnames(num_names)

        else:  # If target is just surname or full name, create surnames.
            surnames = surname_creator.create(num_names, temperature)

    if target != "just_surname":  # If not just surname, forenames must be needed.
        forenames = forename_creators_dict[gender].create(num_names, temperature)

    if target in ["remix", "full_name"]:  # Concat into full name by empty space.
        creations = [f"{forename} {surname}" for forename, surname in zip(forenames, surnames)]

    else:  # One of two lists must be empty if target is just surname or forename.
        creations = surnames + forenames

    end = time.time()
    total_time, avg_time = round(end - start, 2), round((end - start) / num_names, 2)
    return creations, total_time, avg_time
