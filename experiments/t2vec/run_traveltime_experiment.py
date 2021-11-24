from experiments.common.traveltime_estimation import run_traveltime_estimation_experiment

if __name__ == '__main__':
    run_traveltime_estimation_experiment(
        query_result_file='../../data/t2vec-traveltime/traveltime.embeddings.pkl',
        target_file='../../data/t2vec-traveltime/traveltime.durations.pkl'
    )
