from experiments.common.traveltime_estimation import run_traveltime_estimation_experiment

if __name__ == '__main__':
    for rate in [0.0, 0.2, 0.4, 0.6]:
        run_traveltime_estimation_experiment(
            query_result_file=f"../../data/processed_roberta_h1/geohash.test-tte-ds_{rate}.dataframe.results.pkl",
            target_file="../../data/geohash_2/geohash.test-duration.dataframe.pkl",
        )
