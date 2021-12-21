import h5py
import pandas as pd


def process_queries(query_db_pyh5_file: str, destinations_file: str):
    num_db = 10_000

    t2vec_exp1_trj = h5py.File(query_db_pyh5_file)

    destinations = [t2vec_exp1_trj['db']['trips'][str(idx)][-1] for idx in range(1, num_db + 1)]
    traj_destinations = pd.Series(destinations)
    traj_destinations.to_pickle(destinations_file, compression="gzip")


if __name__ == '__main__':
    process_queries(
        query_db_pyh5_file='../../data/t2vec_model_output/exp2-r0-querydb.h5',
        destinations_file='../../data/processed_t2vec/dp.destinations.pkl'
    )
