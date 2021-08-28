"""
ported from https://github.com/boathit/t2vec (./experiments/utils.jl)

"""

import pickle

import numpy as np
import pandas as pd


def uniform_split(trip: np.array):
    return trip[::2], trip[1::2]


def create_query_db(
    trajectory_file: str,
    start: int,
    query_size: int,
    db_size: int,
    query_noise,
    db_noise,
    do_split=True,
    query_db_file="",
    min_length=30,
    max_length=100
):
    n_query, n_db = 0, 0
    trajectory_df = pd.read_pickle(trajectory_file)
    query_df = pd.DataFrame(columns=['trip', 'name'])
    db_df = pd.DataFrame(columns=['trip', 'name'])

    query_filename = f"{query_db_file}.query.dataframe.pkl"
    db_filename = f"{query_db_file}.db.dataframe.pkl"
    meta_filename = f"{query_db_file}.metadata.pkl"
    for i in range(start, len(trajectory_df)):
        trip = trajectory_df.at[i, 'POLYLINE']
        if n_query < query_size:
            if 2 * min_length <= trip.shape[1] <= 2 * max_length:
                if do_split:
                    n_query += 1
                    n_db += 1
                    trip1, trip2 = uniform_split(trip)
                    query_df.loc[i] = [query_noise(trip1), i]
                    db_df.loc[i] = [db_noise(trip1), i]
                else:
                    n_query += 1
                    query_df.loc[i] = [query_noise(trip), i]
        elif n_db < db_size:
            if 2 * min_length <= trip.shape[1] <= 2 * max_length:
                if do_split:
                    n_db += 1
                    trip1, _ = uniform_split(trip)
                    db_df.loc[i] = [db_noise(trip1), i]
                else:
                    n_db += 1
                    db_df.loc[i] = [db_noise(trip), i]
    metadata = {"n_query": n_query, "n_db": n_db}
    query_df.to_pickle(query_filename)
    db_df.to_pickle(db_filename)
    pickle.dump(metadata, open(meta_filename, "wb"))


def euclidean(mat1, mat2):
    return np.linalg.norm(mat1 - mat2)


def rank_search(query_df: pd.DataFrame, db_df: pd.DataFrame):
    def rank(query_trip, query_label):
        dists = db_df['trip'].map(lambda trip: euclidean(query_trip, trip))
        for idx in dists.argsort():
            if query_df.at[idx, 'name'] == query_label:
                return idx + 1

    ranks = []
    for index, row in query_df.iterrows():
        ranks.append(rank(row['trip'], row['name']))
    return ranks
