from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets_impl.taxi_porto import collate_fn_porto, \
    FrameEncodedPortoTaxiTrainDataset, \
    FrameEncodedPortoTaxiValDataset


def run_experiment(
    model: pl.LightningModule,
    gpus: Optional[Union[List[int], str, int]] = 1,
    path_prefix='../data/train-trajectory2vec-v3',
    out_prefix=None
):
    name = model.__class__.__name__

    if out_prefix is None:
        out_prefix = path_prefix

    train_dataset = FrameEncodedPortoTaxiTrainDataset(f'{out_prefix}.train.dataframe.pkl')
    val_dataset = FrameEncodedPortoTaxiValDataset(f'{out_prefix}.val.dataframe.pkl')
    train_loader = DataLoader(train_dataset, batch_size=512, num_workers=10, collate_fn=collate_fn_porto, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, num_workers=10, collate_fn=collate_fn_porto)

    # training
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=600,
        logger=TensorBoardLogger(save_dir="../logs", name=name),
        callbacks=[EarlyStopping(monitor="val_loss", patience=50)]
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer
