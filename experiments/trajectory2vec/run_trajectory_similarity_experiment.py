from experiments.common.trajectory_similarity import run_trajectory_similarity_experiment

if __name__ == '__main__':
    print("Experiment: Subtrajectory")
    for dbsize in [0, 20_000, 40_000, 60_000, 100_000]:
        run_trajectory_similarity_experiment(
            query_results_file="../../data/trajectory2vec-show_timestamps_2/trajectory2vec.test.query.results.pkl",
            db_results_file="../../data/trajectory2vec-show_timestamps_2/trajectory2vec.test.query_database.results.pkl",
            m=dbsize
        )

    print("Experiment: Downsampling")
    for rate in [0.2, 0.4, 0.6]:
        run_trajectory_similarity_experiment(
            query_results_file=
            "../../data/trajectory2vec-show_timestamps_2/trajectory2vec.test-similarity-ds_0.0.dataframe.results.pkl",
            db_results_file=
            f"../../data/trajectory2vec-show_timestamps_2/trajectory2vec.test-similarity-ds_{rate}.dataframe.results.pkl",
            m=0
        )
