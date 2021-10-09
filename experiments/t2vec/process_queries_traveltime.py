import pickle

import h5py
import numpy as np
import pandas as pd


def process_queries(query_db_pyh5_file: str, trj_pyh5_file: str, embeddings_file: str, targets_file: str):
    num_db = 10_000

    t2vec_exp1_trj = h5py.File(trj_pyh5_file)
    db = np.array(t2vec_exp1_trj['layer3'])
    db = [[idx, vec] for idx, vec in enumerate(db)]

    t2vec_exp1_trj = h5py.File(query_db_pyh5_file)

    # sampling frequency = 15 seconds
    # time = 15 * ( n_samples - 1 )
    sizes = [15 * (t2vec_exp1_trj['db']['trips'][str(idx)].shape[0] - 1) for idx in range(1, num_db + 1)]
    travel_durations = pd.Series(sizes)
    travel_durations.to_pickle(targets_file, compression="gzip")

    pickle.dump(list(db), open(embeddings_file, "wb"))


if __name__ == '__main__':
    process_queries(
        query_db_pyh5_file='../../data/t2vec-traveltime/traveltime-querydb.h5',
        trj_pyh5_file='../../data/t2vec-traveltime/traveltime-trj.h5',
        embeddings_file='../../data/t2vec-traveltime/traveltime.embeddings.pkl',
        targets_file='../../data/t2vec-traveltime/traveltime.durations.pkl'
    )
