from experiments.common.trajectory_similarity import run_trajectory_similarity_experiment

if __name__ == '__main__':
    for dbsize in [0, 20_000, 40_000, 60_000, 100_000]:
        run_trajectory_similarity_experiment(
            query_results_file="../../data/train-transformer-h4.test.query.results.pkl",
            db_results_file="../../data/train-transformer-h4.test.query_database.results.pkl",
            m=dbsize
        )
