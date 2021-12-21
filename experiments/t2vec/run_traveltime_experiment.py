from experiments.common.traveltime_estimation import run_traveltime_estimation_experiment

if __name__ == '__main__':
    for rate in [0.0, 0.2, 0.4, 0.6]:
        rate10 = int(rate * 10)
        run_traveltime_estimation_experiment(
            query_result_file=f"../../data/processed_t2vec/exp2-r{rate10}.query.results.pkl",
            target_file="../../data/processed_t2vec/tte.durations.pkl"
        )
