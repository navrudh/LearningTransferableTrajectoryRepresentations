import os
import pickle
from typing import List, Tuple

import h5py
import numpy as np


def process_queries(
    query_db_pyh5_file: str, query_results_file: str, db_results_file: str, num_db: int, num_query: int
):
    t2vec_exp1_trj = h5py.File(query_db_pyh5_file)
    vecs = np.array(t2vec_exp1_trj['layer3'])
    query, db = vecs[:num_query], vecs[num_query:]

    assert num_db == len(db), f"Database size, expected {num_db} was {len(db)}"

    query = [[idx, vec] for idx, vec in enumerate(query)]
    db = [[idx, vec] for idx, vec in enumerate(db)]
    pickle.dump(list(query), open(query_results_file, "wb"))
    pickle.dump(list(db), open(db_results_file, "wb"))


if __name__ == '__main__':
    experiment_dir = '../../data/t2vec_model_output'

    output_dir = '../../data/processed_t2vec'
    os.makedirs(output_dir, exist_ok=True)

    experiment_prefixes: List[Tuple[str, int, int]] = [
        ("exp1", 100_000, 10_000), ("exp2-r0", 10_000, 10_000), ("exp2-r2", 10_000, 10_000),
        ("exp2-r4", 10_000, 10_000), ("exp2-r6", 10_000, 10_000)
    ]

    for exp, num_db, num_query in experiment_prefixes:
        process_queries(
            query_db_pyh5_file=f"{experiment_dir}/{exp}-trj.h5",
            query_results_file=f"{output_dir}/{exp}.query.results.pkl",
            db_results_file=f"{output_dir}/{exp}.db.results.pkl",
            num_db=num_db,
            num_query=num_query,
        )
