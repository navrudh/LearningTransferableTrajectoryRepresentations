import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset


class FrameEncodedPortoTaxiDataset(Dataset):
    def __init__(self, pickle_file):
        self.porto_df = pd.read_pickle(os.path.realpath(pickle_file))

    def __len__(self):
        return len(self.porto_df)

    def __getitem__(self, idx):
        return torch.from_numpy(np.copy(self.porto_df.iloc[idx])), idx


def collate_fn_porto(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    trajectories, drivers = zip(*data)
    lengths = torch.tensor([t.shape[0] for t in trajectories])
    return pack_sequence(trajectories,
                         enforce_sorted=False).float(), pack_sequence(trajectories,
                                                                      enforce_sorted=False).float(), lengths


class FrameEncodedPortoTaxiDatasetValidation(Dataset):
    def __init__(self, pickle_file):
        self.porto_df = pd.read_pickle(os.path.realpath(pickle_file))

    def __len__(self):
        return len(self.porto_df)

    def __getitem__(self, idx):
        return torch.from_numpy(np.copy(self.porto_df.iloc[idx, 1])), \
               torch.from_numpy(np.copy(self.porto_df.iloc[idx, 2])), \
               torch.from_numpy(np.copy(self.porto_df.iloc[idx, 3]))


def collate_fn_porto_val(data):
    originals, gauss, downsampled = zip(*data)

    return pack_sequence(originals, enforce_sorted=False).float(), \
           pack_sequence(gauss, enforce_sorted=False).float(), \
           pack_sequence(downsampled, enforce_sorted=False).float()
