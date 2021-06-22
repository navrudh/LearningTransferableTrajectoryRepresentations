import torch


def packed_sequence_elementwise_apply(fn, packed_sequence):
    """applies fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)
