import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from datasets.trembr.porto_embedding import PortoIterableDataset


class Traj2VecModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Traj2VecModeler, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * embedding_dim, 128)
        self.linear_neighbour = nn.Linear(128, 1)
        self.linear_same_type = nn.Linear(128, 1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, 2 * self.embedding_dim))
        out = F.relu(self.linear1(embeds))
        is_neighbour = torch.sigmoid(self.linear_neighbour(out))
        is_same_type = torch.sigmoid(self.linear_same_type(out))
        return is_neighbour, is_same_type


accelerator = Accelerator()
device = accelerator.device

model = Traj2VecModeler(200000, 256)
optim = torch.optim.Adam(model.parameters())

dataset = PortoIterableDataset()
data = DataLoader(dataset, batch_size=64)

model, optim, data = accelerator.prepare(model, optim, data)

model.train()
for epoch in range(10):
    loss = 0
    for source, tgt_neigh, tgt_type in data:
        optim.zero_grad()

        out_neigh, out_type = model(source)
        loss_neigh = F.binary_cross_entropy(out_neigh, tgt_neigh.unsqueeze(1).float())
        loss_type = F.binary_cross_entropy(out_type, tgt_type.unsqueeze(1).float())
        loss = loss_neigh + loss_type

        accelerator.backward(loss)

        optim.step()

    print(f'Epoch {epoch + 0:03}: | Loss: {loss:.5f}')

torch.save(model.embeddings.state_dict(), "embedding.pt")
