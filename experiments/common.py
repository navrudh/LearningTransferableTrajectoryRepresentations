from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.taxi_porto import collate_fn_porto, FrameEncodedPortoTaxiDataset


def run_experiment(
    model: pl.LightningModule, gpus: Optional[Union[List[int], str, int]] = 1, name: Union[str, None] = None
):
    if name is None:
        name = model.__class__.__name__

    train_dataset = FrameEncodedPortoTaxiDataset('../data/train-trajectory2vec.train.dataframe.pkl')
    val_dataset = FrameEncodedPortoTaxiDataset('../data/train-trajectory2vec.val.dataframe.pkl')
    train_loader = DataLoader(train_dataset, batch_size=2048, collate_fn=collate_fn_porto, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=2048, collate_fn=collate_fn_porto, num_workers=10)

    # training
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=2000,
        logger=TensorBoardLogger(save_dir="../logs", name=name),
        callbacks=[EarlyStopping(monitor="val_loss", patience=200)]
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer
