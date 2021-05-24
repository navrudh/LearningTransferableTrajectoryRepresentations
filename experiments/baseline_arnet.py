import math

import pytorch_lightning as pl
import torch.optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from datasets.taxi_porto import collate_fn_porto, FrameEncodedPortoTaxiDataset
from models.arnet import ARNet


class BaselineArnetExperiment(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ARNet(input_size=35, hidden_size=128, num_classes=500)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat, x_hat_rec, y_pred = self.forward(x)

        reconstruction_loss = F.mse_loss(x_hat, x_hat_rec)
        classification_loss = F.cross_entropy(y_pred, y)
        loss = reconstruction_loss + classification_loss

        return loss


if __name__ == '__main__':
    # data
    dataset = FrameEncodedPortoTaxiDataset('../data/train-preprocessed-taxi.pkl')
    train_size = int(math.ceil(len(dataset) * 0.7))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_porto)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn_porto)

    # model
    model = BaselineArnetExperiment()

    # training
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, train_loader, val_loader)
