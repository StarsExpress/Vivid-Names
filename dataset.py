import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder


class NamesDataset(Dataset):
    def __init__(self, names, max_len):
        self.names = names
        self.max_len = max_len
        self.encoder = LabelEncoder()
        self.encoder.fit(list("".join(names)))
        self.encoded_names = [self.encode_name(name) for name in names]

    def encode_name(self, name):
        encoded = self.encoder.transform(list(name))
        return np.pad(encoded, (0, self.max_len - len(encoded)), 'constant')

    def __len__(self):
        return len(self.encoded_names)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_names[idx], dtype=torch.long)
