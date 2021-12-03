from experiments.common.destination_prediction import run_destination_prediction_experiment

if __name__ == '__main__':
    run_destination_prediction_experiment(
        query_result_file=
        "../../data/trajectory2vec-show_timestamps_2/trajectory2vec.test-dp-traj-ds_0.0.dataframe.results.pkl",
        target_file="../../data/trajectory2vec-show_timestamps_2/trajectory2vec.test-destinations.dataframe.pkl",
        convert_to_gps=True
    )
