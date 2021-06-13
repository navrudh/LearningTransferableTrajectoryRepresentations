import math

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from datasets.taxi_porto import FrameEncodedPortoTaxiDataset, collate_fn_porto


def run_experiment(model: pl.LightningModule):
    # data
    dataset = FrameEncodedPortoTaxiDataset('../data/train-preprocessed-taxi.pkl')
    train_size = int(math.ceil(len(dataset) * 0.7))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=collate_fn_porto, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1024, collate_fn=collate_fn_porto, num_workers=8)

    # training
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, train_loader, val_loader)
