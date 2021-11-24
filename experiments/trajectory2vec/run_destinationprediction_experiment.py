from experiments.common.destination_prediction import run_destination_prediction_experiment

if __name__ == '__main__':
    run_destination_prediction_experiment(
        query_result_file=
        "../../data/train-trajectory2vec-v3-destination-prediction.test-trajectories-processed.embeddings.pkl",
        target_file="../../data/train-trajectory2vec-v3-destination-prediction.test-destinations.dataframe.pkl"
    )
