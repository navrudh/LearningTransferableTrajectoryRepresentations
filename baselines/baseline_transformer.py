import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from datasets_impl.taxi_porto import collate_fn_with_padding
from common import run_experiment


class Transformer(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_size, nhead=2), num_layers=3)
        # self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=256, nhead=4), num_layers=3)
        self.fc = nn.Linear(input_size, 256)

    def forward(self, x):
        b, s, d = x.shape
        # print('x', x.shape)
        # # mask = self._generate_square_subsequent_mask(b)
        # x_enc = self.encoder(x)
        # print('enc', x_enc.shape)
        # x_dec = self.decoder(x_enc)
        # print('dec', x_dec.shape)
        # # out = self.fc(x_enc)
        x_enc = self.encoder(x)
        print(x_enc.reshape((b, -1)).shape)
        out = self.fc(x_enc.reshape((b, -1)))
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        src, _, tgt, _ = train_batch
        src = self.forward(src)
        tgt = self.forward(tgt)
        # print(src.shape)
        # print(tgt.shape)
        loss = F.mse_loss(src, tgt)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        src, _, tgt, _ = val_batch
        src = self.forward(src)
        tgt = self.forward(tgt)
        # print(src.shape)
        # print(tgt.shape)
        loss = F.mse_loss(src, tgt)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)


if __name__ == '__main__':
    model = Transformer(input_size=2)
    trainer = run_experiment(
        model=model, collate_fn=collate_fn_with_padding, gpus=[0], path_prefix="../data/raw-gps-trajectories"
    )
    trainer.save_checkpoint("../data/transformer-raw-gps.ckpt")
