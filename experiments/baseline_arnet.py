import pytorch_lightning as pl
import torch.optim
from torch.nn import functional as F

from experiments.common import run_experiment
from models.arnet import ARNetReconstructionLoss


class BaselineArnetExperiment(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ARNetReconstructionLoss(input_size=35, hidden_size=128, num_classes=500)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat, x_hat_rec = self.forward(x)

        reconstruction_loss = F.mse_loss(x_hat, x_hat_rec)

        return reconstruction_loss


if __name__ == '__main__':
    run_experiment(model=BaselineArnetExperiment())
