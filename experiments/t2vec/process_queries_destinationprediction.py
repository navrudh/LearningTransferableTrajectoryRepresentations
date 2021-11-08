import pickle

import h5py
import numpy as np
import pandas as pd


def process_queries(query_db_pyh5_file: str, trj_pyh5_file: str, embeddings_file: str, destinations_file: str):
    num_db = 10_000

    t2vec_exp1_trj = h5py.File(trj_pyh5_file)
    db = np.array(t2vec_exp1_trj['layer3'])
    db = [[idx, vec] for idx, vec in enumerate(db)]

    t2vec_exp1_trj = h5py.File(query_db_pyh5_file)

    destinations = [t2vec_exp1_trj['db']['trips'][str(idx)][-1] for idx in range(1, num_db + 1)]
    traj_destinations = pd.Series(destinations)
    traj_destinations.to_pickle(destinations_file, compression="gzip")

    pickle.dump(list(db), open(embeddings_file, "wb"))


if __name__ == '__main__':
    process_queries(
        query_db_pyh5_file='../../data/t2vec-traveltime/traveltime-querydb.h5',
        trj_pyh5_file='../../data/t2vec-traveltime/traveltime-trj.h5',
        embeddings_file='../../data/t2vec-destination-prediction/test-trajectories.embeddings.pkl',
        destinations_file='../../data/t2vec-destination-prediction/test-destinations.dataframe.pkl'
    )
