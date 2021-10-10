from experiments.trajectory2vec.process_queries import load_eval_model, process_queries

if __name__ == '__main__':
    eval_model = load_eval_model(path="../../data/trajectory2vec-traveltime.ckpt", input_size=30)

    for rate in [0.0, 0.2, 0.4, 0.6]:
        process_queries(
            query_file=f"../../data/train-trajectory2vec-v3-no-timesteps.test-ds-{rate}.query_database.pkl",
            results_file=f"../../data/train-trajectory2vec-v3-no-timesteps.test-ds-{rate}.query_database.results.pkl",
            eval_model=eval_model
        )
