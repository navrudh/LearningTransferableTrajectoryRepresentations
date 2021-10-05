import pickle

import h5py
import numpy as np

if __name__ == '__main__':
    t2vec_exp1_trj = h5py.File('../../data/t2vec-simulated/clustering-trj.h5')
    vecs = np.array(t2vec_exp1_trj['layer3'])
    valid_vecs = [vec for vec in vecs if not np.isnan(vec).any()]
    print(len(valid_vecs))
    pickle.dump(valid_vecs, open('../../data/t2vec-simulated/simulated_test_1000_clustering-t2vec.pkl', "wb"))
