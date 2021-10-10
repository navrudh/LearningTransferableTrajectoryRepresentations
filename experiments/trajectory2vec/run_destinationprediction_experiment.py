import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pl_bolts.datamodules import SklearnDataset
from sklearn.model_selection import KFold
from torch.optim import SGD
from torch.utils.data import DataLoader

from models.linear_regression import LinearRegression


def run_experiment(query_result_file: str, target_file: str):
    data = pickle.load(open(query_result_file, 'rb'))
    data = [item[1] for item in data]
    data = np.stack(data, axis=0)
    target = pd.read_pickle(os.path.realpath(target_file), compression="gzip")
    target = np.stack(target.to_list(), axis=0).astype(float)

    # convert meters to kilometers
    target = target / 1000.

    kfold = KFold(n_splits=5, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
        for idx, _ in enumerate(['latitude', 'longitude']):
            train_dataset = SklearnDataset(X=data[train_idx], y=target[train_idx], y_transform=lambda val: val[idx])
            test_dataset = SklearnDataset(X=data[test_idx], y=target[test_idx], y_transform=lambda val: val[idx])
            train_loader = DataLoader(train_dataset, num_workers=2)
            test_loader = DataLoader(test_dataset, num_workers=2)
            model = LinearRegression(input_dim=256, learning_rate=1e-2, optimizer=SGD)
            trainer = pl.Trainer(gpus=1, max_epochs=2, deterministic=True)
            trainer.fit(model, train_dataloader=train_loader)
            trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    pl.seed_everything(42)

    run_experiment(
        query_result_file=
        "../../data/train-trajectory2vec-v3-destination-prediction.test-trajectories-processed.embeddings.pkl",
        target_file="../../data/train-trajectory2vec-v3-destination-prediction.test-destinations.dataframe.pkl"
    )
