import pickle

import numpy as np
import pandas
from tqdm import tqdm


def sqeuclidean(v1, v2):
    d = v1 - v2
    return np.dot(d, d)


def subtrajectory_experiment(query_results_file: str, query_database_results_file: str, m: int = 0):
    db_size = m + 10_000
    query_results = pickle.load(open(query_results_file, "rb"))
    query_database_results = pickle.load(open(query_database_results_file, "rb"))
    query_database_results = query_database_results[:db_size]
    query_database = pandas.DataFrame(query_database_results, columns=["TRAJECTORY_ID", "VECTOR"])

    ranks = []
    for _, (query_id, query_vector) in enumerate(
            tqdm(query_results, desc=f"searching query (m= {m}, dbsize={len(query_database_results)})")
    ):
        query_database['DISTANCE'] = query_database["VECTOR"].transform(lambda vec: sqeuclidean(query_vector, vec))
        sorted_database = query_database.sort_values(by=['DISTANCE'])

        # ranks.append(sorted_database[sorted_database['TRAJECTORY_ID'] == query_id].index.values[0] + 1)
        for (idx, trajectory_id, _, _) in sorted_database.itertuples():
            if trajectory_id == query_id:
                ranks.append(idx + 1)
                break

    mean_rank = np.mean(ranks)
    print("Database Len:", len(query_database))
    print("M:", m)
    print("Mean Rank:", mean_rank)


if __name__ == '__main__':
    for dbsize in [20000, 40000, 60000, 80000, 100000]:
        subtrajectory_experiment(
            query_results_file="../../data/train-trajectory2vec.test.query.results.pkl",
            query_database_results_file="../../data/train-trajectory2vec.test.query_database.results.pkl",
            m=dbsize
        )
