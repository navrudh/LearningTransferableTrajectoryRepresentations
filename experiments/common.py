from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets.taxi_porto import collate_fn_porto_train, collate_fn_porto_val, \
    FrameEncodedPortoTaxiTrainDataset, \
    FrameEncodedPortoTaxiValDataset


def run_experiment(
    model: pl.LightningModule,
    gpus: Optional[Union[List[int], str, int]] = 1,
    path_prefix='../data/train-trajectory2vec-v3'
):
    name = model.__class__.__name__

    train_dataset = FrameEncodedPortoTaxiTrainDataset(f'{path_prefix}.train.dataframe.pkl')
    val_dataset = FrameEncodedPortoTaxiValDataset(f'{path_prefix}.val.dataframe.pkl')
    train_loader = DataLoader(
        train_dataset, batch_size=2048, num_workers=10, collate_fn=collate_fn_porto_train, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=2048, num_workers=10, collate_fn=collate_fn_porto_val)

    # training
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=600,
        logger=TensorBoardLogger(save_dir="../logs", name=name),
        callbacks=[EarlyStopping(monitor="val_loss", patience=50)]
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer
