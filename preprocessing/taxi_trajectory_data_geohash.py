"""

Preparing transformer data similar to t2vec, to run identical experiments

script version: v1

"""

import csv
import json
import random
from pathlib import Path
from typing import List, Tuple, Union

import geohash2
import modin.pandas as pd
import numpy as np

from preprocessing.common import downsampling_distort, get_database_file, get_dataset_file, get_query_file, \
    get_subtrajectories, panda_types, save_pickle
from utils.array import downsampling
from utils.gps import lonlat2meters, meters2lonlat


def save_csv(series: pd.Series, outfile: Union[str, Path]):
    series.to_csv(
        str(outfile) + ".csv.gz",
        compression="gzip",
        sep=' ',
        index=False,
        header=False,
        quotechar="",
        quoting=csv.QUOTE_NONE,
        escapechar=" "
    )

    print("Saved: " + str(outfile) + ".csv.gz")


def prepare_taxi_data(in_file: str, out_prefix: str):
    train_prefix = out_prefix + ".train"
    val_prefix = out_prefix + ".val"
    test_prefix = out_prefix + ".test"

    file_df = pd.read_csv(in_file, dtype=panda_types, usecols=['POLYLINE'])

    print("Cleaning Dataset")

    file_df['POLYLINE'] = file_df['POLYLINE'].apply(lambda s: json.loads(s))
    file_df = file_df[file_df['POLYLINE'].map(len) >= 20]
    file_df = file_df[file_df['POLYLINE'].map(len) <= 100]
    file_df['POLYLINE'] = file_df['POLYLINE'].apply(lambda gps_list: gps2meters(gps_list))

    # shuffle and split
    train_size = 800_000
    test_size = 10_000
    val_size = 10_000
    train_df = file_df[:train_size]
    test_df = file_df[train_size:]

    print("Processing Train")

    # create source data
    distorted = train_df['POLYLINE'].apply(lambda gps_meter_list: downsampling_distort(gps_meter_list))
    distorted = distorted.explode()
    distorted = distorted.apply(lambda trajectory: meters2gps(trajectory))
    distorted = distorted.apply(lambda gps_trajectory: geohashify(gps_trajectory))

    distorted = distorted.apply(lambda geohash_list: " ".join(geohash_list))
    save_csv(distorted, get_dataset_file(train_prefix))

    print("Processing Test")
    test_series = test_df['POLYLINE'].apply(lambda trajectory: meters2gps(trajectory))
    test_series = test_series.apply(lambda gps_trajectory: geohashify(gps_trajectory))

    # val
    print("Generate: Validation")
    val_series = test_series.sample(n=val_size)
    test_series = test_series.loc[~test_series.index.isin(val_series.index)]
    # validation prepared exactly like training data

    val_series = val_series.apply(lambda geohash_list: " ".join(geohash_list))
    save_csv(val_series, get_dataset_file(val_prefix))

    # experiment queries

    print("Experiment: TRAJECTORY SIMILARITY")
    # q - query trajectories
    # p - additional traj from the test set
    qp = test_series.sample(n=100_000 + test_size)
    q = qp[:test_size]
    p = qp[test_size:]

    q_a, q_b = get_subtrajectories(q)
    p_a, p_b = get_subtrajectories(p)
    query_db = pd.concat([q_b, p_a])  # p_b is ignored as we need only 100_000 traj

    q_a = q_a.apply(lambda geohash_list: " ".join(geohash_list))
    save_csv(q_a, get_query_file(test_prefix))

    query_db = query_db.apply(lambda geohash_list: " ".join(geohash_list))
    save_csv(query_db, get_database_file(test_prefix))

    print("Experiment: DESTINATION PREDICTION")
    destination_task_data = test_series.sample(n=test_size)
    destinations = destination_task_data.apply(lambda trip: geohash2.decode_exactly(trip[-1]))
    save_pickle(destinations, get_dataset_file(test_prefix, suffix="destinations"))

    destination_task_trajectories = destination_task_data.apply(lambda trip: trip[:int(len(trip) * 0.8)])
    destination_task_trajectories = destination_task_trajectories.apply(lambda geohash_list: " ".join(geohash_list))
    save_csv(destination_task_trajectories, get_dataset_file(test_prefix, suffix="dp-trajectories"))

    print("Experiment: TRAVEL-TIME ESTIMATION")
    traveltime_task_data = test_series.sample(n=test_size)
    travel_durations = traveltime_task_data.apply(lambda trip: len(trip) * 15)
    save_pickle(travel_durations, get_dataset_file(test_prefix, suffix="duration"))

    traveltime_trajectories_raw = traveltime_task_data.apply(lambda geohash_list: " ".join(geohash_list))
    save_csv(traveltime_trajectories_raw, get_dataset_file(test_prefix, suffix="tte-ds_0.0"))
    for rate in [0.2, 0.4, 0.6]:
        print("downsampling rate : " + str(rate))
        downsampled_trajectories = traveltime_task_data.apply(lambda trip: downsampling(np.array(trip), rate).tolist())
        downsampled_trajectories = downsampled_trajectories.apply(lambda geohash_list: " ".join(geohash_list))
        save_csv(downsampled_trajectories, get_dataset_file(test_prefix, suffix=f"tte-ds_{rate}"))


def gps2meters(polyline: List):
    if polyline is None or len(polyline) == 0:
        return []

    arr = np.array(polyline, dtype=np.float32)
    x, y = lonlat2meters(arr[:, 0], arr[:, 1])
    return np.vstack((x, y)).T


def meters2gps(trajectory: np.array):
    if trajectory is None or len(trajectory) == 0:
        return []

    lat, lon = meters2lonlat(trajectory[:, 0], trajectory[:, 1])
    return np.vstack((lat, lon)).T.tolist()


def geohashify(trajectory: List[Tuple]):
    return [geohash2.encode(pt[1], pt[0], 7) for pt in iter(trajectory)]


if __name__ == '__main__':
    random.seed(49)
    np.random.seed(49)

    import ray

    ray.shutdown()
    ray.init()

    prepare_taxi_data(in_file="../data/train.csv", out_prefix="../data/train-transformer")
