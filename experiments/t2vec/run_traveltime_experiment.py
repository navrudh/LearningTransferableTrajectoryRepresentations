import pytorch_lightning as pl

from experiments.trajectory2vec.run_traveltime_experiment import run_experiment

if __name__ == '__main__':
    pl.seed_everything(42)

    run_experiment(
        query_result_file='../../data/t2vec-traveltime/traveltime.embeddings.pkl',
        target_file='../../data/t2vec-traveltime/traveltime.durations.pkl'
    )
