import pandas as pd
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset


class FrameEncodedPortoTaxiDataset(Dataset):
    def __init__(self, pickle_file):
        self.porto_df = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.porto_df)

    def __getitem__(self, idx):
        return torch.from_numpy(self.porto_df.iloc[idx, 1]), self.porto_df.iloc[idx, 0]


def collate_fn_porto(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    trajectories, drivers = zip(*data)
    return pack_sequence(trajectories, enforce_sorted=False).float(), torch.tensor(drivers)
