import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.taxi_porto import FrameEncodedPortoTaxiValDataset
from experiments.baseline_trajectory2vec import BaselineTrajectory2VecExperiment


def load_eval_model(path: str):
    model = BaselineTrajectory2VecExperiment.load_from_checkpoint(path)
    model.double()
    model.eval()
    return model


def process_test(query_file, results_file, eval_model: BaselineTrajectory2VecExperiment):
    """
    Save the output of the last hidden layer

    :param query_file: queries extracted from the test set
    :param results_file: file to save the hidden state
    :param eval_model: the model that processes queries to results
    :return: None
    """
    dataset = FrameEncodedPortoTaxiValDataset(query_file)
    loader = DataLoader(dataset)

    test_results = []

    for _, (sample, idx) in enumerate(tqdm(loader, desc="running query")):
        hidden = eval_model.forward(sample.double(), None, is_train=False)
        hidden = hidden.detach().numpy()[-1, -1]
        test_results.append([idx.item() + 1, hidden])

    pickle.dump(test_results, open(results_file, "wb"))


if __name__ == '__main__':
    eval_model = load_eval_model(path="../../data/trajectory2vec-v3.ckpt")
    process_test(
        query_file="../../data/train-trajectory2vec-v3.test.query.pkl",
        results_file="../../data/train-trajectory2vec-v3.test.query.results.pkl",
        eval_model=eval_model
    )
    process_test(
        query_file="../../data/train-trajectory2vec-v3.test.query_database.pkl",
        results_file="../../data/train-trajectory2vec-v3.test.query_database.results.pkl",
        eval_model=eval_model
    )
