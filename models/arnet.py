"""
Paper: Autoencoder Regularized Network For Driving Style Representation Learning

Reference: arXiv:1701.01272v1
"""

import torch.nn


class ARNetReconstructionLoss(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, batch_first=True, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.gru_layers = torch.nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=batch_first
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(in_features=hidden_size, out_features=64)
        self.fc2 = torch.nn.Linear(64, hidden_size)
        # self.fc3 = torch.nn.Linear(in_features=hidden_size, out_features=num_classes)
        # self.reconstruction_loss = torch.nn.MSELoss()
        # self.classification_loss = torch.nn.CrossEntropyLoss()
        self.relu = torch.nn.ReLU(inplace=True)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # b, t, f = x.shape
        _, h_n = self.gru_layers(x)
        h_last_layer = h_n.view(-1, 2, self.hidden_size)[:, -1, :]
        x_hat: torch.Tensor = self.dropout(h_last_layer)
        rec: torch.Tensor = self.tanh(self.fc2(self.relu(self.fc1(x_hat))))
        # y_pred: torch.Tensor = self.fc3(x_hat)
        return x_hat, rec  # , y_pred
