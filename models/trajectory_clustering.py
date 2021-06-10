"""
Paper: Learning deep representation for trajectory clustering

Link: DOI:10.1111/exsy.12252
"""

import torch.nn


class AutoEncoderModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, batch_first=True, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.encoder = torch.nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=batch_first
        )
        self.decoder = torch.nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=batch_first
        )

    def forward(self, x):
        # b, t, f = x.shape
        _, h_n = self.encoder(x)
        h_last_layer = h_n.view(-1, 2, self.hidden_size)[:, -1, :]
        x_hat: torch.Tensor = self.decoder(h_last_layer).view(-1, 2, self.hidden_size)[:, -1, :]
        return x_hat
