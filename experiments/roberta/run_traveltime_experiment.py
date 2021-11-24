from experiments.common.traveltime_estimation import run_traveltime_estimation_experiment

if __name__ == '__main__':
    for rate in [0.6]:  # , 0.2, 0.4, 0.6]:
        print(f"train-transformer.test-ds-{rate}.query_database.results.pkl")
        run_traveltime_estimation_experiment(
            query_result_file=f"../../data/train-transformer.test-ds-{rate}.query_database.results.pkl",
            target_file="../../data/train-transformer.test-duration.dataframe.pkl"
        )
