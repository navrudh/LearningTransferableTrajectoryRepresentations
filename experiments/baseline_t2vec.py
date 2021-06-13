import pytorch_lightning as pl
import torch.optim
from torch.nn import functional as F

from experiments.common import run_experiment
from models.t2vec import EncoderDecoder


class BaselineT2VecExperiment(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EncoderDecoder(
            vocab_size=18864, embedding_size=256, hidden_size=256, num_layers=3, dropout=0.2, bidirectional=True
        )

    def forward(self, src, lengths, tgt):
        return self.model.forward(src, lengths, tgt)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, lengths, y = train_batch
        x_hat, x_hat_rec = self.forward(x, lengths, x.detach().clone())

        reconstruction_loss = F.mse_loss(x_hat, x_hat_rec)

        return reconstruction_loss


if __name__ == '__main__':
    run_experiment(model=BaselineT2VecExperiment())
