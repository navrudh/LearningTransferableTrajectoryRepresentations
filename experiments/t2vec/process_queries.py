import pickle

import h5py
import numpy as np


def process_queries(query_db_pyh5_file: str, query_results_file: str, db_results_file: str):
    num_query = 1000
    num_db = 100_000

    t2vec_exp1_trj = h5py.File(query_db_pyh5_file)
    vecs = np.array(t2vec_exp1_trj['layer3'])
    query, db = vecs[:num_query], vecs[num_query:]
    query = [[idx, vec] for idx, vec in enumerate(query)]
    db = [[idx, vec] for idx, vec in enumerate(db)]
    pickle.dump(list(query), open(query_results_file, "wb"))
    pickle.dump(list(db), open(db_results_file, "wb"))


if __name__ == '__main__':
    process_queries(
        query_db_pyh5_file='../../data/t2vec/exp1-trj.h5',
        query_results_file='../../data/t2vec/query.results.pkl',
        db_results_file='../../data/t2vec/query_database.results.pkl'
    )
