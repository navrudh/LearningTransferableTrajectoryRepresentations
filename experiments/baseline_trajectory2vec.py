from typing import List, Tuple

import pytorch_lightning as pl
import torch.optim
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from experiments.common import run_experiment
from models.trajectory2vec import EncoderDecoder


class BaselineTrajectory2VecExperiment(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EncoderDecoder(input_size=30, hidden_size=256, num_layers=2, dropout=0.2, bidirectional=False)

    def forward(self, src, tgt, is_train: bool):
        return self.model.forward(src, tgt, is_train)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: Tuple[PackedSequence, PackedSequence, List[int]], batch_idx):
        src, tgt, lengths = train_batch
        x_hat = self.forward(src, tgt, is_train=True)
        x = pad_packed_sequence(src)[0]
        loss = F.mse_loss(x, x_hat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        src, tgt, lengths = val_batch
        x_hat = self.forward(src, tgt, is_train=True)
        x = pad_packed_sequence(src)[0]
        loss = F.mse_loss(x, x_hat)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)


if __name__ == '__main__':
    model = BaselineTrajectory2VecExperiment()
    trainer = run_experiment(model=model, gpus=[0])
    trainer.save_checkpoint("trajectory2vec.ckpt")
