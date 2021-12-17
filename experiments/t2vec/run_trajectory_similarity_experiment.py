from experiments.common.trajectory_similarity import run_trajectory_similarity_experiment

if __name__ == '__main__':
    experiment_dir = '../../data/processed_t2vec'

    for dbsize in [0, 20_000, 40_000, 60_000, 100_000]:
        run_trajectory_similarity_experiment(
            query_results_file=f"{experiment_dir}/exp1.query.results.pkl",
            db_results_file=f"{experiment_dir}/exp1.db.results.pkl",
            m=dbsize
        )

    # print("Experiment: Downsampling")
    # for rate in [0.0, 0.2, 0.4, 0.6]:
    #     rate10 = int(rate * 10)
    #     run_trajectory_similarity_experiment(
    #         query_results_file=f"{experiment_dir}/exp2-r{rate10}.query.results.pkl",
    #         db_results_file=f"{experiment_dir}/exp2-r0.db.results.pkl",
    #         m=0
    #     )
