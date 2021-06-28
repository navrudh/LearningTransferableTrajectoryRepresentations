from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.taxi_porto import collate_fn_porto, collate_fn_porto_val, FrameEncodedPortoTaxiDataset, \
    FrameEncodedPortoTaxiDatasetValidation


def run_experiment(
    model: pl.LightningModule, gpus: Optional[Union[List[int], str, int]] = 1, name: Union[str, None] = None
):
    if name is None:
        name = model.__class__.__name__

    train_dataset = FrameEncodedPortoTaxiDataset('../data/train-preprocessed-taxi.pkl')
    val_dataset = FrameEncodedPortoTaxiDatasetValidation('../data/val-preprocessed-taxi.pkl')
    train_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=collate_fn_porto, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1024, collate_fn=collate_fn_porto_val, num_workers=8)

    # training
    trainer = pl.Trainer(gpus=gpus, max_epochs=10, logger=TensorBoardLogger(save_dir="../logs", name=name))
    trainer.fit(model, train_loader, val_loader)
