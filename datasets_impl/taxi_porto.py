import os

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset


class FrameEncodedPortoTaxiValDataset(Dataset):
    def __init__(self, pickle_file):
        self.porto_df = pd.read_pickle(os.path.realpath(pickle_file), compression="gzip")

    def __len__(self):
        return len(self.porto_df)

    def __getitem__(self, idx):
        return torch.from_numpy(np.copy(self.porto_df.iloc[idx])), \
               torch.from_numpy(np.copy(self.porto_df.iloc[idx]))


class FrameEncodedTrajectoryDatasetWithIndex(Dataset):
    def __init__(self, pickle_file):
        self.porto_df = pd.read_pickle(os.path.realpath(pickle_file), compression="gzip")

    def __len__(self):
        return len(self.porto_df)

    def __getitem__(self, idx):
        return torch.from_numpy(np.copy(self.porto_df.iloc[idx])), idx


class FrameEncodedPortoTaxiTrainDataset(Dataset):
    def __init__(self, pickle_file):
        self.porto_df = pd.read_pickle(os.path.realpath(pickle_file), compression="gzip")

    def __len__(self):
        return len(self.porto_df)

    def __getitem__(self, idx):
        return torch.from_numpy(np.copy(self.porto_df.iloc[idx, 1])), \
               torch.from_numpy(np.copy(self.porto_df.iloc[idx, 0]))


def collate_fn_porto(data):
    """
       data: is a list of tuples with (source, target)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    sources, targets = zip(*data)
    return pack_sequence(sources, enforce_sorted=False).float(), \
           pack_sequence(targets, enforce_sorted=False).float()


def collate_fn_with_padding(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths), 2).float()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # # sort a list by sequence length (descending order) to use pack_padded_sequence
    # data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths


# class FrameEncodedPortoTaxiDatasetValidation(Dataset):
#     def __init__(self, pickle_file):
#         self.porto_df = pd.read_pickle(os.path.realpath(pickle_file))
#
#     def __len__(self):
#         return len(self.porto_df)
#
#     def __getitem__(self, idx):
#         return torch.from_numpy(np.copy(self.porto_df.iloc[idx, 1])), \
#                torch.from_numpy(np.copy(self.porto_df.iloc[idx, 2])), \
#                torch.from_numpy(np.copy(self.porto_df.iloc[idx, 3]))
#
#
# def collate_fn_porto_val(data):
#     originals, gauss, downsampled = zip(*data)
#
#     return pack_sequence(originals, enforce_sorted=False).float(), \
#            pack_sequence(gauss, enforce_sorted=False).float(), \
#            pack_sequence(downsampled, enforce_sorted=False).float()
