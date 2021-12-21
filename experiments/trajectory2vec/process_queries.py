import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.baseline_trajectory2vec import BaselineTrajectory2VecExperiment
from datasets_impl.taxi_porto import FrameEncodedTrajectoryDatasetWithIndex


def load_eval_model(path: str, **kwargs):
    model = BaselineTrajectory2VecExperiment.load_from_checkpoint(path, map_location=torch.device("cuda:0"), **kwargs)
    model.double()
    model.eval()
    return model


def process_queries(query_file, results_file, eval_model: BaselineTrajectory2VecExperiment):
    """
    Save the output of the last hidden layer

    :param query_file: queries extracted from the test set
    :param results_file: file to save the hidden state
    :param eval_model: the model that processes queries to results
    :return: None
    """
    print("query_file:", query_file)

    dataset = FrameEncodedTrajectoryDatasetWithIndex(query_file)
    loader = DataLoader(dataset)

    test_results = []

    for _, (sample, idx) in enumerate(tqdm(loader, desc="running query")):
        hidden = eval_model.forward(sample.double(), None, is_train=False)
        hidden = hidden.detach().numpy()[-1, -1]
        test_results.append([idx.item() + 1, hidden])

    pickle.dump(test_results, open(results_file, "wb"))


if __name__ == '__main__':
    eval_model = load_eval_model(path=f"../../data/models/trajectory2vec.ckpt", input_size=36)

    experiment_dir = "../../data/"

    output_dir = '../../data/processed_trajectory2vec'
    os.makedirs(output_dir, exist_ok=True)

    input_files = [
        # destination prediction
        "trajectory2vec.test-dp-traj-ds_0.0.dataframe.pkl",
        "trajectory2vec.test-dp-traj-ds_0.2.dataframe.pkl",
        "trajectory2vec.test-dp-traj-ds_0.4.dataframe.pkl",
        "trajectory2vec.test-dp-traj-ds_0.6.dataframe.pkl",
        # travel time estimation
        "trajectory2vec.test-tte-ds_0.0.dataframe.pkl",
        "trajectory2vec.test-tte-ds_0.2.dataframe.pkl",
        "trajectory2vec.test-tte-ds_0.4.dataframe.pkl",
        "trajectory2vec.test-tte-ds_0.6.dataframe.pkl",
        # similarity
        "trajectory2vec.test.query.pkl",
        "trajectory2vec.test-similarity-ds_0.2.dataframe.pkl",
        "trajectory2vec.test-similarity-ds_0.4.dataframe.pkl",
        "trajectory2vec.test-similarity-ds_0.6.dataframe.pkl",
        "trajectory2vec.test.dataframe.pkl",
    ]

    for src_file in input_files:
        dest_file = src_file.replace(".pkl", ".results.pkl")
        process_queries(
            query_file=f"{experiment_dir}/{src_file}", results_file=f"{output_dir}/{dest_file}", eval_model=eval_model
        )
