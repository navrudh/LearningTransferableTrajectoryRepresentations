from experiments.common.destination_prediction import run_destination_prediction_experiment

if __name__ == '__main__':
    run_destination_prediction_experiment(
        query_result_file="../../data/train-transformer.test-dest-traj.query.results.pkl",
        target_file="../../data/train-transformer.test-destinations.dataframe.pkl"
    )
