from experiments.experiment_subtrajectories import subtrajectory_experiment

if __name__ == '__main__':
    for dbsize in [0]:
        subtrajectory_experiment(
            query_results_file="../../data/train-transformer.test.query.results.pkl",
            db_results_file="../../data/train-transformer.test.query_database.results.pkl",
            m=dbsize
        )
