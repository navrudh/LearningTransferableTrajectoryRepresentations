from experiments.trajectory2vec.process_queries import load_eval_model, process_queries

if __name__ == '__main__':
    eval_model = load_eval_model(path="../../data/trajectory2vec-v3.ckpt", input_size=36)
    process_queries(
        query_file="../../data/train-trajectory2vec-v3-destination-prediction.test-trajectories-processed.dataframe.pkl",
        results_file=
        "../../data/train-trajectory2vec-v3-destination-prediction.test-trajectories-processed.embeddings.pkl",
        eval_model=eval_model
    )
