from experiments.common.destination_prediction import run_destination_prediction_experiment

if __name__ == '__main__':
    for rate in [0.0, 0.2, 0.4, 0.6]:
        run_destination_prediction_experiment(
            query_result_file=f"../../data/processed_roberta_h1/geohash.test-dp-traj-ds_{rate}.dataframe.results.pkl",
            target_file="../../data/geohash_2/geohash.test-destinations.dataframe.pkl",
        )
