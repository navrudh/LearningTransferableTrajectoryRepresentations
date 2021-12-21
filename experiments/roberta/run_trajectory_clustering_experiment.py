from experiments.common.trajectory_clustering import get_embeddings_from_results_pkl, visualize_clustering_tsne

if __name__ == '__main__':
    embeddings = get_embeddings_from_results_pkl(
        "../../data/simulated/processed_roberta/geohash_test.train.dataframe.results.pkl"
    )
    visualize_clustering_tsne(embeddings, group_size=1000)
