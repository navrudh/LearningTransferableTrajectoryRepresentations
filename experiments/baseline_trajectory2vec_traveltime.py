from experiments.baseline_trajectory2vec import BaselineTrajectory2VecExperiment
from experiments.common import run_experiment

if __name__ == '__main__':
    model = BaselineTrajectory2VecExperiment(input_size=30)
    trainer = run_experiment(model=model, gpus=[1], path_prefix='../data/train-trajectory2vec-v3-no-timesteps')
    trainer.save_checkpoint("../data/trajectory2vec-traveltime.ckpt")
