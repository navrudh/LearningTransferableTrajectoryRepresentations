"""
Paper: Learning deep representation for trajectory clustering

Reference: https://doi.org/10.1111/exsy.12252
"""

from typing import Tuple

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        """
        embedding (vocab_size, input_size): pretrained embedding
        """
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size, self.hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout
        )

    def forward(self, input: PackedSequence, h0=None) -> Tuple[PackedSequence, Tensor]:
        output, hn = self.rnn(input, h0)
        return output, hn


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(Decoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size, self.hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout
        )

    def forward(self, input, h):
        output, hn = self.rnn(input, h)
        return output, hn


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.fc = nn.Linear(hidden_size, input_size)
        self.num_layers = num_layers

    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0) // 2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size) \
                .transpose(1, 2).contiguous() \
                .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, trg, is_train=True):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        trg (trg_seq_len, batch): target tensor, the `seq_len` in trg is not
            necessarily the same as that in src
        ---
        Output:
        output (trg_seq_len, batch, hidden_size)
        """
        encoder_output, encoder_hn = self.encoder(src)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)

        if not is_train:
            return decoder_h0

        output, decoder_hn = self.decoder(trg, decoder_h0)
        output = pad_packed_sequence(output)[0]
        output = self.fc(output)
        return output
