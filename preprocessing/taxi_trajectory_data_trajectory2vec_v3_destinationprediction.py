"""

Preparing trajectory2vec data similar to t2vec, to run identical experiments

script version: v3

"""

from preprocessing.taxi_trajectory_data_trajectory2vec_v3 import *


def prepare_taxi_data(in_file: str, out_prefix: str, seq_len=600, window_len=300):
    train_prefix = out_prefix + ".train"
    val_prefix = out_prefix + "-destination-prediction" + ".val"
    test_prefix = out_prefix + "-destination-prediction" + ".test"

    file_df = pd.read_csv(in_file, dtype=panda_types, usecols=['POLYLINE'])

    file_df['POLYLINE'] = file_df['POLYLINE'].transform(lambda s: json.loads(s))
    file_df = file_df[file_df['POLYLINE'].map(len) >= 20]
    file_df = file_df[file_df['POLYLINE'].map(len) <= 100]
    file_df['POLYLINE'] = file_df['POLYLINE'].transform(lambda gps_list: gps2meters(gps_list))

    # shuffle and split
    train_size = 800_000
    train_df = file_df[:train_size]
    test_df = file_df[train_size:]

    # normalization
    metadata = pickle.load(open(get_metadata_file(train_prefix), "rb"))
    tr_min = metadata["train_min"]
    tr_max = metadata["train_max"]
    tr_diff = tr_max - tr_min

    # val
    print("build validation data")
    val_df = test_df.sample(n=10_000)
    test_df = test_df.loc[~test_df.index.isin(val_df.index)]

    print('build test data')
    d = test_df.sample(n=10_000)

    destinations = d['POLYLINE'].apply(lambda trip: trip[-1])
    destinations.to_pickle(get_dataset_file(test_prefix, suffix="destinations"), compression="gzip")

    d['POLYLINE'] = d['POLYLINE'].apply(lambda trip: trip[:int(len(trip) * 0.8)])
    d['POLYLINE'].to_pickle(get_dataset_file(test_prefix, suffix="trajectories"), compression="gzip")

    queries = build_behavior_matrix(d['POLYLINE'], seq_len, window_len, tr_min, tr_max, tr_diff)
    queries.to_pickle(get_dataset_file(test_prefix, suffix="trajectories-processed"), compression="gzip")


if __name__ == '__main__':
    random.seed(49)
    np.random.seed(49)

    import ray

    ray.shutdown()
    ray.init()

    prepare_taxi_data(
        in_file="../data/train.csv", out_prefix="../data/train-trajectory2vec-v3", seq_len=300, window_len=150
    )
