"""

Source taken from the pl_bolts repository

"""

from typing import Dict, List, Tuple, Type

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


class LinearRegression(pl.LightningModule):
    """
    Linear regression model implementing - with optional L1/L2 regularization
    $$min_{W} ||(Wx + b) - y ||_2^2 $$
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        bias: bool = True,
        learning_rate: float = 1e-3,
        optimizer: Type[Optimizer] = Adam,
        scheduler=None,
        scheduler_kwargs=None,
        scheduler_config=None,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
    ) -> None:
        """
        Args:
            input_dim: number of dimensions of the input (1+)
            output_dim: number of dimensions of the output (default: ``1``)
            bias: If false, will not use $+b$
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default: ``Adam``)
            l1_strength: L1 regularization strength (default: ``0.0``)
            l2_strength: L2 regularization strength (default: ``0.0``)
        """
        super().__init__()
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.scheduler:
            assert scheduler_kwargs is not None, 'parameter scheduler_kwargs should not be None'
            assert scheduler_config is not None, 'parameter scheduler_config should not be None'
            self.scheduler_kwargs = scheduler_kwargs
            self.scheduler_config = scheduler_config

        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.linear(x)
        return y_hat

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch

        y_hat = self(x)

        loss = F.l1_loss(y_hat, y)

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg

        tensorboard_logs = {"train_mae_loss": loss}
        progress_bar_metrics = tensorboard_logs
        return {"loss": loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        y_hat = self(x)
        return {"test_loss": F.l1_loss(y_hat, y)}

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_mae_loss": test_loss}
        progress_bar_metrics = tensorboard_logs
        print(progress_bar_metrics)
        return {"test_loss": test_loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def configure_optimizers(self):
        if self.scheduler is not None:
            optim = self.optimizer(self.parameters(), lr=self.hparams.learning_rate)
            sched = {'scheduler': self.scheduler(optim, **self.scheduler_kwargs), **self.scheduler_config}
            return [optim], [sched]
        else:
            return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)
