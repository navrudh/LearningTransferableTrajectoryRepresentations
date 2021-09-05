import pickle

from torch.utils.data import DataLoader

from datasets.taxi_porto import FrameEncodedPortoTaxiDataset
from experiments.baseline_trajectory2vec import BaselineTrajectory2VecExperiment


def load_eval_model(path: str):
    model = BaselineTrajectory2VecExperiment.load_from_checkpoint(path)
    model.double()
    model.eval()
    return model


def process_test(query_file, results_file, eval_model: BaselineTrajectory2VecExperiment):
    dataset = FrameEncodedPortoTaxiDataset(query_file)
    loader = DataLoader(dataset)

    test_results = []

    for _, (sample, idx) in enumerate(loader):
        hidden = eval_model.forward(sample.double(), None, is_train=False)
        hidden = hidden.detach().numpy()[-1, -1]
        test_results.append([idx, hidden])

    pickle.dump(test_results, open(results_file, "wb"))


if __name__ == '__main__':
    eval_model = load_eval_model(path="../../data/trajectory2vec.ckpt")
    process_test(
        query_file="../../data/sample-trajectory2vec.test.query.pkl",
        results_file="../../data/sample-trajectory2vec.test.query.results.pkl",
        eval_model=eval_model
    )
    process_test(
        query_file="../../data/sample-trajectory2vec.test.query_database.pkl",
        results_file="../../data/sample-trajectory2vec.test.query_database.results.pkl",
        eval_model=eval_model
    )