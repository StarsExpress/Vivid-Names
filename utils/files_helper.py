from configs.paths_config import DATA_FOLDER_PATH
from vae.dataset import NamesDataset
import os
import pandas as pd
import json
import pickle


timesteps_path = os.path.join(DATA_FOLDER_PATH, f"timesteps.json")


def read_unique_names(name_type: str):
    """
    Read unique names from a specified file.

    If names aren't unique, remove duplicates and save unique names back to the read file.

    Args:
        name_type (str): type of names to read. Can be: 'male_forenames', 'female_forenames', 'surnames'.

    Returns:
        list: list of unique names.
    """
    file_path = os.path.join(
        DATA_FOLDER_PATH, f'{name_type}.txt'
    )
    names_df = pd.read_csv(file_path, sep='\t', header=None, names=['names'])

    if names_df['names'].is_unique is False:  # Update file.
        names_df.drop_duplicates(inplace=True)
        names_df.reset_index(drop=True, inplace=True)
        names_df.to_csv(file_path, header=False, index=False)
    return names_df['names'].tolist()


def update_timesteps(partial_dict: dict[str, int]):
    """
    Update timesteps.json file with a partial dict carrying a specific name type's timesteps.

    Args:
        partial_dict (dict[str, int]): key is name type and value is timesteps.
    """
    with open(timesteps_path, "r") as file:
        timesteps_dict = json.load(file)

    timesteps_dict.update(partial_dict)

    with open(timesteps_path, "w") as file:
        json.dump(timesteps_dict, file)


def read_timesteps(name_type: str):
    """
    Read timesteps of a specific name type from timesteps.json file.

    Args:
        name_type (str): type of name to read timesteps for.

    Returns:
        int: timesteps of specified type of name.
    """
    with open(timesteps_path, "r") as file:
        timesteps_dict = json.load(file)
    return timesteps_dict[name_type]


def save_dataset(name_type: str, dataset: NamesDataset):
    """
    Save NamesDataset object to a pickle file.

    Args:
        name_type (str): type of names to corresponding dataset.
        dataset (NamesDataset): dataset to be saved.
    """
    file_path = os.path.join(
        DATA_FOLDER_PATH, f'{name_type}.pkl'
    )
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)


def read_dataset(name_type: str):
    """
    Read NamesDataset object from a pickle file.

    Args:
        name_type (str): type of names to corresponding dataset.

    Returns:
        NamesDataset: dataset read from file.
    """
    file_path = os.path.join(
        DATA_FOLDER_PATH, f'{name_type}.pkl'
    )
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset
