import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np


class NamesDataset(Dataset):

    def __init__(self, names: list, max_len: int):
        self.names = names
        self.max_len = max_len

        self.encoder = LabelEncoder()
        self.encoder.fit(list("".join(names)))

        self.encoded_names = [self.encode_name(name) for name in names]

    def encode_name(self, name: str):
        encoded_name = self.encoder.transform(list(name))
        return np.pad(
            encoded_name,
            (0, self.max_len - len(encoded_name)),
            'constant'
        )

    def __len__(self):
        return len(self.encoded_names)

    def __getitem__(self, idx: int):
        return torch.tensor(self.encoded_names[idx], dtype=torch.long)
