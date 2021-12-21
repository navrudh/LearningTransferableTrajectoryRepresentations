import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def get_embeddings_from_results_pkl(file):
    records = pickle.load(open(file, 'rb'))
    return np.array([row[1] for row in records])


def get_embeddings_from_flat_pkl(file):
    records = pickle.load(open(file, 'rb'))
    return np.array([row for row in records])


def visualize_clustering_tsne(embeddings, group_size=100):
    labels = ['Straight', 'Circular', 'Bending']
    color_codes = [0] * group_size + [1] * group_size + [2] * group_size

    fig, ax = plt.subplots()

    tsne_components = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(embeddings).T
    scatter = ax.scatter(tsne_components[0], tsne_components[1], c=color_codes)

    legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=labels, title="Trajectory Shapes")
    ax.add_artist(legend1)
    plt.show()
