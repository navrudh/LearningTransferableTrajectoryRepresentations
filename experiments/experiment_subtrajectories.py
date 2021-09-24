import pickle

import numpy as np
from tqdm import tqdm


def sqeuclidean(v1, v2):
    d = v1 - v2
    return np.dot(d, d)


def subtrajectory_experiment(query_results_file: str, db_results_file: str, m: int = 0):
    db_size = m + 10_000
    query_results = pickle.load(open(query_results_file, "rb"))
    query_database_results = pickle.load(open(db_results_file, "rb"))
    query_database_results = query_database_results[:db_size]
    # query_database = pandas.DataFrame(query_database_results, columns=["TRAJECTORY_ID", "VECTOR"])

    ranks = []
    for _, (query_id, query_vector) in enumerate(
        tqdm(query_results, desc=f"searching query (m= {m}, dbsize={len(query_database_results)})")
    ):
        # query_database['DISTANCE'] = query_database["VECTOR"].transform(lambda vec: sqeuclidean(query_vector, vec))
        # sorted_database = query_database.sort_values(by=['DISTANCE'])

        sorted_db = [db_result + [sqeuclidean(query_vector, db_result[-1])] for db_result in query_database_results]
        sorted_db.sort(key=lambda entry: entry[-1])

        # ranks.append(sorted_database[sorted_database['TRAJECTORY_ID'] == query_id].index.values[0] + 1)
        # for (idx, trajectory_id, _, _) in sorted_database.itertuples():
        for idx, (trajectory_id, _, _) in enumerate(sorted_db):
            if trajectory_id == query_id:
                ranks.append(idx + 1)
                break

    mean_rank = np.mean(ranks)
    print("Database Len:", len(query_database_results))
    print("M:", m)
    print("Mean Rank:", mean_rank)


if __name__ == '__main__':
    for dbsize in [0]:
        subtrajectory_experiment(
            query_results_file='../../data/t2vec/query.results.pkl',
            db_results_file='../../data/t2vec/query_database.results.pkl',
            m=dbsize
        )
