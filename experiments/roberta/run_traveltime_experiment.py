import pytorch_lightning as pl

from experiments.trajectory2vec.run_traveltime_experiment import run_experiment

if __name__ == '__main__':
    pl.seed_everything(42)

    for rate in [0.6]:  # , 0.2, 0.4, 0.6]:
        print(f"train-transformer.test-ds-{rate}.query_database.results.pkl")
        run_experiment(
            query_result_file=f"../../data/train-transformer.test-ds-{rate}.query_database.results.pkl",
            target_file="../../data/train-transformer.test-duration.dataframe.pkl"
        )
