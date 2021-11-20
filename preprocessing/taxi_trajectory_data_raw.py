"""

Preparing trajectory2vec data similar to t2vec, to run identical experiments

script version: v3

"""

import json
import random

import modin.pandas as pd
import numpy as np

from preprocessing.taxi_trajectory_data_geohash import downsampling, downsampling_distort, get_database_file, \
    get_dataset_file, get_query_file, get_subtrajectories, gps2meters, meters2gps, panda_types, save_csv


def prepare_taxi_data(in_file: str, out_prefix: str):
    train_prefix = out_prefix + ".train"
    val_prefix = out_prefix + ".val"
    test_prefix = out_prefix + ".test"

    file_df = pd.read_csv(in_file, dtype=panda_types, usecols=['POLYLINE'])

    file_df['POLYLINE'] = file_df['POLYLINE'].apply(lambda s: json.loads(s))
    file_df = file_df[file_df['POLYLINE'].map(len) >= 20]
    file_df = file_df[file_df['POLYLINE'].map(len) <= 100]
    file_df['POLYLINE'] = file_df['POLYLINE'].apply(lambda gps_list: gps2meters(gps_list))

    # shuffle and split
    train_size = 800_000
    train_df = file_df[:train_size]
    test_df = file_df[train_size:]

    print("Processing Train")

    # create source data
    distorted = train_df['POLYLINE'].apply(lambda gps_meter_list: downsampling_distort(gps_meter_list))
    distorted = distorted.explode()
    distorted = distorted.apply(lambda trajectory: meters2gps(trajectory))
    save_csv(distorted, get_dataset_file(train_prefix))

    print("Processing Test")
    test_series = test_df['POLYLINE'].apply(lambda trajectory: meters2gps(trajectory))

    # val
    print("Generate: Validation")
    val_series = test_series.sample(n=10_000)
    test_series = test_series.loc[~test_series.index.isin(val_series.index)]
    # validation prepared exactly like training data

    save_csv(val_series, get_dataset_file(val_prefix))

    # experiment queries

    print("Experiment: TRAJECTORY SIMILARITY")
    # q - query trajectories
    # p - additional traj from the test set
    qp = test_series.sample(n=100_000 + 10_000)
    q = qp[:10_000]
    p = qp[10_000:]

    q_a, q_b = get_subtrajectories(q)
    p_a, p_b = get_subtrajectories(p)
    query_db = pd.concat([q_b, p_a])  # p_b is ignored as we need only 100_000 traj

    save_csv(q_a, get_query_file(test_prefix))

    save_csv(query_db, get_database_file(test_prefix))

    print("Experiment: DESTINATION PREDICTION")
    destination_task_data = test_series.sample(n=10_000)
    destinations = destination_task_data.apply(lambda trip: trip[-1])
    destinations.to_pickle(get_dataset_file(test_prefix, suffix="destinations"), compression="gzip")

    destination_task_trajectories = destination_task_data.apply(lambda trip: trip[:int(len(trip) * 0.8)])
    save_csv(destination_task_trajectories, get_query_file(test_prefix + "-dest-traj"))

    print("Experiment: TRAVEL-TIME ESTIMATION")
    traveltime_task_data = test_series.sample(n=10_000)
    travel_durations = traveltime_task_data.apply(lambda trip: len(trip) * 15)
    travel_durations.to_pickle(get_dataset_file(test_prefix, suffix="duration"), compression="gzip")

    save_csv(traveltime_task_data, get_database_file(test_prefix + "-ds-0.0"))
    for rate in [0.2, 0.4, 0.6]:
        print("downsampling rate : " + str(rate))
        downsampled_trajectories = traveltime_task_data.apply(lambda trip: downsampling(np.array(trip), rate).tolist())
        save_csv(downsampled_trajectories, get_database_file(test_prefix + "-ds-" + str(rate)))


if __name__ == '__main__':
    random.seed(49)
    np.random.seed(49)

    import ray

    ray.shutdown()
    ray.init()

    prepare_taxi_data(
        in_file="../data/train.csv",
        out_prefix="../data/raw-gps-trajectories",
    )
