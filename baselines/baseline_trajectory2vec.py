from typing import Tuple

import pytorch_lightning as pl
import torch.optim
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from baselines.common import run_experiment
from models.trajectory2vec import EncoderDecoder


class BaselineTrajectory2VecExperiment(pl.LightningModule):
    def __init__(self, input_size: int):
        super().__init__()
        self.model = EncoderDecoder(
            input_size=input_size, hidden_size=256, num_layers=3, dropout=0.2, bidirectional=False
        )

    def forward(self, src, tgt, is_train: bool):
        return self.model.forward(src, tgt, is_train)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch: Tuple[PackedSequence, PackedSequence], batch_idx):
        src, tgt = train_batch
        x_hat = self.forward(src, tgt, is_train=True)
        x = pad_packed_sequence(tgt)[0]
        loss = F.mse_loss(x, x_hat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: Tuple[PackedSequence, PackedSequence], batch_idx):
        src, tgt = val_batch
        x_hat = self.forward(src, tgt, is_train=True)
        x = pad_packed_sequence(tgt)[0]
        loss = F.mse_loss(x, x_hat)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)


if __name__ == '__main__':
    model = BaselineTrajectory2VecExperiment(input_size=36)
    trainer = run_experiment(model=model, gpus=[1], path_prefix='../data/trajectory2vec-show-timestamp/trajectory2vec')
    trainer.save_checkpoint("../data/models/trajectory2vec-show-timestamp.ckpt")
