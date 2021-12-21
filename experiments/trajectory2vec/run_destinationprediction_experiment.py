from experiments.common.destination_prediction import run_destination_prediction_experiment

if __name__ == '__main__':
    for rate in [0.0, 0.2, 0.4, 0.6]:
        run_destination_prediction_experiment(
            query_result_file=
            f"../../data/processed_trajectory2vec/trajectory2vec.test-dp-traj-ds_{rate}.dataframe.results.pkl",
            target_file="../../data/trajectory2vec-show_timestamps_3/trajectory2vec.test-destinations.dataframe.pkl",
            convert_to_gps=True
        )
