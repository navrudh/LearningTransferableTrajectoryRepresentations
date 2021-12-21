import h5py
import numpy as np

from experiments.common.trajectory_clustering import visualize_clustering_tsne

if __name__ == '__main__':
    t2vec_exp_file = h5py.File("../../data/simulated/t2vec/clustering-trj.h5")
    embeddings = np.array(t2vec_exp_file['layer3'])
    visualize_clustering_tsne(embeddings, group_size=1000)
