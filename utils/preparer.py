import pandas as pd


def prepare_csv(gender: str, names: list[str]):
    """
    Given created names and gender, return CSV.

    Args:
        gender (str): gender of names. Used as header.
        names (list[str]): list of names to put into CSV.

    Returns:
        bytes: encoded CSV.
    """
    names_df = pd.DataFrame(names, columns=[f'{gender.capitalize()} Names'])
    return names_df.to_csv(index=False).encode('utf-8')


def prepare_txt(gender: str, names: list[str]):
    """
    Given created names and gender, return text file.

    Args:
        gender (str): gender of names. Used as header.
        names (list[str]): list of names to turn into text.

    Returns:
        str: text file of names.
    """
    return '\n'.join([f'{gender.capitalize()} Names'] + names)
