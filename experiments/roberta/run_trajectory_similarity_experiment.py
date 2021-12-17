from experiments.common.trajectory_similarity import run_trajectory_similarity_experiment

if __name__ == '__main__':
    experiment_dir: str = "../../data/processed_roberta_h8"

    print("Experiment: Expanded Search-space")
    for dbsize in [0, 20_000, 40_000, 60_000, 100_000]:
        run_trajectory_similarity_experiment(
            query_results_file=f"{experiment_dir}/geohash.test.query.results.pkl",
            db_results_file=f"{experiment_dir}/geohash.test.dataframe.results.pkl",
            m=dbsize
        )

    print("Experiment: Downsampling")
    for rate in [0.0, 0.2, 0.4, 0.6]:
        run_trajectory_similarity_experiment(
            query_results_file=f"{experiment_dir}/geohash.test-similarity-ds_0.0.dataframe.results.pkl",
            db_results_file=f"{experiment_dir}/geohash.test-similarity-ds_{rate}.dataframe.results.pkl",
            m=0
        )
