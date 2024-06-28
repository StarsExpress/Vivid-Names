from configs.app_config import DATA_FOLDER_PATH
from vae.dataset import NamesDataset
import os
import pandas as pd
import json
import pickle


max_len_path = os.path.join(DATA_FOLDER_PATH, 'max_len.json')


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


def update_max_len(max_len: dict[str, int]):
    with open(max_len_path, "r") as file:
        max_len_dict = json.load(file)

    max_len_dict.update(max_len)

    with open(max_len_path, "w") as file:
        json.dump(max_len_dict, file)


def read_max_len(name_type: str):
    with open(max_len_path, "r") as file:
        max_len_dict = json.load(file)
    return max_len_dict[name_type]


def save_dataset(name_type: str, dataset: NamesDataset):
    file_path = os.path.join(
        DATA_FOLDER_PATH, f'{name_type}.pkl'
    )
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)


def read_dataset(name_type: str):
    file_path = os.path.join(
        DATA_FOLDER_PATH, f'{name_type}.pkl'
    )
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset
