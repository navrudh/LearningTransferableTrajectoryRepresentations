import os
import pickle

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

    kfold = KFold(n_splits=5, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
        train_dataset = SklearnDataset(X=data[train_idx], y=target[train_idx])
        test_dataset = SklearnDataset(X=data[test_idx], y=target[test_idx])
        train_loader = DataLoader(train_dataset, num_workers=2)
        test_loader = DataLoader(test_dataset, num_workers=2)
        model = LinearRegression(input_dim=256, learning_rate=1e-3, optimizer=SGD)
        trainer = pl.Trainer(gpus=1, max_epochs=5, deterministic=True)
        trainer.fit(model, train_dataloader=train_loader)
        trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    pl.seed_everything(42)

    for rate in [0.0]:  # , 0.2, 0.4, 0.6]:
        print(f"train-trajectory2vec-v3-no-timesteps.test-ds-{rate}.query_database.results.pkl")
        run_experiment(
            query_result_file=
            f"../../data/train-trajectory2vec-v3-no-timesteps.test-ds-{rate}.query_database.results.pkl",
            target_file="../../data/train-trajectory2vec-v3-no-timesteps.test-duration.dataframe.pkl"
        )
