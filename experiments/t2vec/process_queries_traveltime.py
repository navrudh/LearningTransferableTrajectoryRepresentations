import h5py
import pandas as pd


def extract_durations(query_db_pyh5_file: str, targets_file: str):
    num_db = 10_000

    t2vec_exp1_trj = h5py.File(query_db_pyh5_file)

    # sampling frequency = 15 seconds
    # time = 15 * ( n_samples - 1 )
    sizes = [15 * (t2vec_exp1_trj['db']['trips'][str(idx)].shape[0] - 1) for idx in range(1, num_db + 1)]
    travel_durations = pd.Series(sizes)
    travel_durations.to_pickle(targets_file, compression="gzip")


if __name__ == '__main__':
    extract_durations(
        query_db_pyh5_file='../../data/t2vec_model_output/exp2-r0-querydb.h5',
        targets_file='../../data/processed_t2vec/tte.durations.pkl'
    )
