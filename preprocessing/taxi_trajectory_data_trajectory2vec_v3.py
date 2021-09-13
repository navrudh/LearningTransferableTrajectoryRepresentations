"""

Preparing trajectory2vec data similar to t2vec, to run identical experiments

script version: v3

"""

import json
import math
import pickle
import random
from pathlib import Path
from typing import List

import modin.pandas as pd
import numpy as np

from utils.array import downsampling
from utils.gps import distort, distort_gps_array, downsample_gps_array, lonlat2meters

EPSILON = 1e-10
FLOAT_MAX = np.iinfo(np.int64).max
FLOAT_MIN = np.iinfo(np.int64).min
panda_types = {
    'TRIP_ID': np.uint64,
    'CALL_TYPE': str,
    'ORIGIN_CALL': str,
    'ORIGIN_STAND': str,
    'TAXI_ID': np.uint64,
    'TIMESTAMP': np.uint64,
    'DAY_TYPE': str,
    'MISSING_DATA': bool,
    'POLYLINE': str,
}
DOWNSAMPLING_RATES = [.1, .15, .2, .25, .3]
DATASET_SAMPLE_RATE = 15  # gps coordinated are sampled every 15 seconds


def get_dataset_file(path):
    return Path(f"{path}.dataframe.pkl")


def get_metadata_file(path):
    return Path(f"{path}.metadata.pkl")


def get_database_file(path):
    return Path(f"{path}.query_database.pkl")


def get_query_file(path):
    return Path(f"{path}.query.pkl")


def prepare_taxi_data(in_file: str, out_prefix: str, seq_len=600, window_len=300):
    train_prefix = out_prefix + ".train"
    val_prefix = out_prefix + ".val"
    test_prefix = out_prefix + ".test"

    file_df = pd.read_csv(in_file, dtype=panda_types, usecols=['POLYLINE'])

    metadata = {}

    file_df['POLYLINE'] = file_df['POLYLINE'].transform(lambda s: json.loads(s))
    file_df = file_df[file_df['POLYLINE'].map(len) >= 20]
    file_df = file_df[file_df['POLYLINE'].map(len) <= 100]
    file_df['POLYLINE'] = file_df['POLYLINE'].transform(lambda gps_list: gps2meters(gps_list))

    # shuffle and split
    train_size = 800_000
    train_df = file_df[:train_size]
    test_df = file_df[train_size:]

    train_df['SOURCE'] = train_df['POLYLINE'].transform(lambda gps_meter_list: downsampling_distort(gps_meter_list))
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(
        lambda gps_meter_list: sliding_window(gps_meter_list, seq_len, window_len)
    )
    train_df = train_df.explode('SOURCE')
    train_df['SOURCE'] = train_df['SOURCE'].apply(
        lambda gps_meter_list: sliding_window_varying_samplerate(gps_meter_list, seq_len, window_len)
    )
    # normalization
    tr_min, tr_max = compute_min_max(train_df)
    tr_diff = tr_max - tr_min
    metadata["train_min"] = tr_min
    metadata["train_max"] = tr_max
    pickle.dump(metadata, open(get_metadata_file(train_prefix), "wb"))
    print("normalizing target")
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(lambda arr: (arr - tr_min) / tr_diff)
    print("normalizing source")
    train_df['SOURCE'] = train_df['SOURCE'].transform(lambda arr: (arr - tr_min) / tr_diff)

    print("saving training data")
    train_df.to_pickle(get_dataset_file(train_prefix))

    # val
    print("build validation data")
    val_df = test_df.sample(n=10_000)
    test_df = test_df.loc[~test_df.index.isin(val_df.index)]
    # validation prepared exactly like training data

    val_df['POLYLINE'] = build_behavior_matrix(val_df['POLYLINE'], seq_len, window_len, tr_min, tr_max, tr_diff)
    val_df['POLYLINE'].to_pickle(get_dataset_file(val_prefix))

    # experiment queries
    print("build query set")
    # q - query trajectories
    # p - additional traj from the test set
    qp = test_df.sample(n=100_000 + 10_000)
    q = qp[:10_000]
    p = qp[10_000:]

    q_a, q_b = get_subtrajectories(q)
    p_a, p_b = get_subtrajectories(p)
    query_db = pd.concat([q_b, p_a])  # p_b is ignored as we need only 100_000 traj

    q_a = build_behavior_matrix(q_a, seq_len, window_len, tr_min, tr_max, tr_diff)
    q_a.to_pickle(get_query_file(test_prefix))

    query_db = build_behavior_matrix(query_db, seq_len, window_len, tr_min, tr_max, tr_diff)
    query_db.to_pickle(get_database_file(test_prefix))


def build_behavior_matrix(df, seq_len, window_len, tr_min, tr_max, tr_diff):
    df = df.transform(lambda gps_meter_list: sliding_window(gps_meter_list, seq_len, window_len))
    df = df.transform(lambda arr: (arr - tr_min) / tr_diff)
    return df


def gps2meters(polyline: List):
    if len(polyline) == 0 or polyline is None:
        return None

    arr = np.array(polyline, dtype=np.float32)
    x, y = lonlat2meters(arr[:, 0], arr[:, 1])
    timesteps = np.arange(len(polyline)) * DATASET_SAMPLE_RATE
    return np.vstack((x, y, timesteps)).T


def sliding_window(arr: np.array, window_size_seconds: int, slide_step_seconds: int):
    if arr is None:
        return []

    actual_sample_rate = int(arr[1, 2] - arr[0, 2])
    window_size = window_size_seconds // actual_sample_rate
    slide_step = slide_step_seconds // actual_sample_rate
    output_len = 1 + max(0, int(math.ceil((arr.shape[0] - window_size) / slide_step)))
    """
    make sure windowing does not need padding
    
    """
    movement_features = calc_car_movement_features(arr)

    if movement_features is None:
        return []

    windowed_results = [
        calc_feature_stat_matrix(movement_features[step * slide_step:step * slide_step + window_size])
        for step in range(output_len)
    ]

    return [element for element in windowed_results if element is not None]


def sliding_window_varying_samplerate(arr: np.array, window_size_seconds: int, slide_step_seconds: int):
    movement_features = calc_car_movement_features(arr)

    if movement_features is None:
        return []

    windows = rolling_window(movement_features, window_size_seconds, slide_step_seconds)

    windowed_results = [calc_feature_stat_matrix(np.array(window)) for window in windows]

    return [element for element in windowed_results if element is not None]


def rolling_window(sample, windowsize, offset):
    timeLength = sample[len(sample) - 1][0]
    windowLength = int(timeLength / offset) + 1
    windows = []
    for i in range(0, windowLength):
        windows.append([])

    for record in sample:
        time = record[0]
        for i in range(0, windowLength):
            if (time > (i * offset)) & (time < (i * offset + windowsize)):
                windows[i].append(record)
    return windows


def calc_car_movement_features(arr: np.array):
    if arr.shape[0] <= 2:
        return None

    gps_pos = arr[:, [0, 1]]
    timesteps = arr[:, 2]
    seq_len = gps_pos.shape[0]
    time_diff = timesteps[1:] - timesteps[:-1]

    velocity = np.divide(gps_pos[1:] - gps_pos[:-1], time_diff[:, np.newaxis])
    velocity_norm = np.linalg.norm(gps_pos[1:] - gps_pos[:-1], axis=1) / time_diff
    diff_velocity_norm = velocity_norm[1:] - velocity_norm[:-1]
    acceleration_norm = np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)
    diff_acceleration_norm = acceleration_norm[1:] - acceleration_norm[:-1]
    angular_velocity = np.arctan((gps_pos[1:, 1] - gps_pos[:-1, 1]) / (gps_pos[1:, 0] - gps_pos[:-1, 0] + EPSILON))

    return np.column_stack(
        (
            timesteps,
            np.pad(velocity_norm, (seq_len - velocity_norm.shape[0], 0), 'constant'),
            np.pad(diff_velocity_norm, (seq_len - diff_velocity_norm.shape[0], 0), 'constant'),
            np.pad(acceleration_norm, (seq_len - acceleration_norm.shape[0], 0), 'constant'),
            np.pad(diff_acceleration_norm, (seq_len - diff_acceleration_norm.shape[0], 0), 'constant'),
            np.pad(angular_velocity, (seq_len - angular_velocity.shape[0], 0), 'constant'),
        )
    )


def calc_feature_stat_matrix(x: np.ndarray):
    if x.shape[0] == 0:
        return None

    return np.concatenate(
        (
            x.mean(axis=0),
            x.min(axis=0, initial=FLOAT_MAX),
            np.percentile(x, 25, axis=0),
            np.percentile(x, 50, axis=0),
            np.percentile(x, 75, axis=0),
            x.max(axis=0, initial=FLOAT_MIN),
        )
    )


def compute_mean_std(df: pd.DataFrame):
    _mean = df['POLYLINE'].transform(lambda arr: np.mean(arr, 0, keepdims=True)).to_numpy().mean()
    _std = df['POLYLINE'].transform(lambda arr: np.mean(arr, 0, keepdims=True)).to_numpy().std()
    # df['POLYLINE'] = df['POLYLINE'].transform(lambda arr: (arr - _mean) / _std)
    return _mean, _std


def compute_min_max(df: pd.DataFrame):
    _min = df['POLYLINE'].transform(lambda arr: np.min(arr, 0, initial=FLOAT_MAX)).to_list()
    _max = df['POLYLINE'].transform(lambda arr: np.max(arr, 0, initial=FLOAT_MIN)).to_list()

    _min = np.min(_min, axis=0)
    _max = np.max(_max, axis=0)
    return _min, _max


def normalize_local_metrics(df: pd.DataFrame):
    df['POLYLINE'] = df['POLYLINE'].transform(lambda arr: (arr - np.mean(arr, 0)) / np.std(arr, 0))


def apply_gaussian_noise(lst: List):
    trip = distort_gps_array(lst, .22, 50)
    trip[0] = lst[0]
    trip[-1] = lst[-1]
    return trip


def apply_downsampling(lst: List):
    return downsample_gps_array(lst, rate=DOWNSAMPLING_RATES[np.random.randint(0, 4 + 1)])


def downsampling_distort(trip: np.ndarray):
    noise_trips = []
    dropping_rates = [0, 0.2, 0.4, 0.5, 0.6]
    distorting_rates = [0, 0.2, 0.4, 0.6]
    for dropping_rate in dropping_rates:
        noisetrip1 = downsampling(trip, dropping_rate)
        for distorting_rate in distorting_rates:
            noisetrip2 = distort(noisetrip1, distorting_rate)
            noise_trips.append(noisetrip2)
    return noise_trips


def get_subtrajectories(df: pd.DataFrame):
    df_a = df['POLYLINE'].transform(lambda polyline_list: polyline_list[::2])
    df_b = df['POLYLINE'].transform(lambda polyline_list: polyline_list[1::2])
    return df_a, df_b


if __name__ == '__main__':
    random.seed(49)
    np.random.seed(49)

    import ray

    ray.shutdown()
    ray.init()

    prepare_taxi_data(
        in_file="../data/train.csv", out_prefix="../data/train-trajectory2vec-v2", seq_len=300, window_len=150
    )
