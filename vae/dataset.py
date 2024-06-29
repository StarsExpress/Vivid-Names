import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np


class NamesDataset(Dataset):
    """
    A custom PyTorch Dataset for encoding names.

    Convert a list of names into a format that can be used by a PyTorch model.

    Each name is encoded by a LabelEncoder, and padded to a fixed length.

    Attributes:
        names (list): original list of names.
        max_len (int): max length for a name. All names will be padded to this length.
        encoder (LabelEncoder): encoder that converts names into numerical format.
        encoded_names (list): list of names after encoding and padding.

    Methods:
        encode_name(name: str): encodes a single name via LabelEncoder and pads it to max length.

    Methods:
        encode_name(name: str): encodes a single name via LabelEncoder and pads it to max length.
        __len__(): returns length of encoded names list.
        __getitem__(idx: int): returns encoded name at given index.
    """

    def __init__(self, names: list, max_len: int):
        """
        Initialize dataset with list of names and a max length.

        Args:
            names (list): original list of names.
            max_len (int): max length for a name. All names will be padded to this length.
        """
        self.names = names
        self.max_len = max_len

        self.encoder = LabelEncoder()
        self.encoder.fit(list("".join(names)))

        self.encoded_names = [self.encode_name(name) for name in names]

    def encode_name(self, name: str):
        """
        Encode a single name using LabelEncoder and pad it to max length.

        Args:
            name (str): name to be encoded.

        Returns:
            np.array: encoded and padded name.
        """
        encoded_name = self.encoder.transform(list(name))
        return np.pad(
            encoded_name,
            (0, self.max_len - len(encoded_name)),
            'constant'
        )

    def __len__(self):
        """
        Returns length of encoded names list.
        """
        return len(self.encoded_names)

    def __getitem__(self, idx: int):
        """
        Returns encoded name at given index.

        Args:
            idx (int): index of encoded name to return.

        Returns:
            torch.Tensor: encoded name at given index.
        """
        return torch.tensor(self.encoded_names[idx], dtype=torch.long)
