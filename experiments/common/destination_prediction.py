import os
import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pl_bolts.datamodules import SklearnDataset
from sklearn.model_selection import KFold
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.linear_regression import LinearRegression


def run_destination_prediction_experiment(query_result_file: str, target_file: str):
    pl.seed_everything(42)

    data = pickle.load(open(query_result_file, 'rb'))
    data = [item[1] for item in data]
    data = np.stack(data, axis=0)
    target = pd.read_pickle(os.path.realpath(target_file), compression="gzip")
    target = np.stack(target.to_list(), axis=0).astype(float)
    target = target[:, :2]

    # convert meters to kilometers
    target = target / 1000.

    kfold = KFold(n_splits=5, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
        # train_dataset = SklearnDataset(X=data[train_idx], y=target[train_idx])
        # test_dataset = SklearnDataset(X=data[test_idx], y=target[test_idx])
        # train_loader = DataLoader(train_dataset, num_workers=2)
        # test_loader = DataLoader(test_dataset, num_workers=2)
        # model = LinearRegression(
        #     input_dim=256,
        #     output_dim=2,
        #     learning_rate=1e-2,
        #     optimizer=SGD,
        #     scheduler=ExponentialLR,
        #     scheduler_kwargs={'gamma': 0.85},
        #     scheduler_config={'interval': 'epoch'}
        # )
        # trainer = pl.Trainer(gpus=1, max_epochs=30, deterministic=True)
        # trainer.fit(model, train_dataloader=train_loader)
        # trainer.test(test_dataloaders=test_loader)
        for idx, metric_name in enumerate(['latitude', 'longitude']):
            print(metric_name)
            train_dataset = SklearnDataset(X=data[train_idx], y=target[train_idx], y_transform=lambda val: val[idx])
            test_dataset = SklearnDataset(X=data[test_idx], y=target[test_idx], y_transform=lambda val: val[idx])
            train_loader = DataLoader(train_dataset, num_workers=2)
            test_loader = DataLoader(test_dataset, num_workers=2)
            model = LinearRegression(
                input_dim=256,
                output_dim=1,
                learning_rate=1e-2,
                optimizer=SGD,
                scheduler=ExponentialLR,
                scheduler_kwargs={'gamma': 0.85},
                scheduler_config={'interval': 'epoch'}
            )
            trainer = pl.Trainer(gpus=1, max_epochs=5, deterministic=True, logger=False)
            trainer.fit(model, train_dataloader=train_loader)
            trainer.test(test_dataloaders=test_loader)
