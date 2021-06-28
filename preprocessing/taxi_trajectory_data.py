import json
import math
from pathlib import Path
from typing import List

import modin.pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided

from utils.gps import distort_gps_array, downsample_gps_array

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


def prepare_taxi_data(seq_len=256, window_len=32):
    train_file_path = Path("../data/train-preprocessed-taxi.pkl")
    val_file_path = Path("../data/val-preprocessed-taxi.pkl")
    if train_file_path.exists():
        print("Preprocessed taxi data file already exists")
        return
    file_df = pd.read_csv("../data/train.csv", dtype=panda_types, usecols=['TAXI_ID', 'POLYLINE'])

    # shuffle and split
    file_df.reindex(np.random.permutation(file_df.index))
    train_size = 0.9
    train_end = int(len(file_df) * train_size)
    train_df = file_df[:train_end]
    val_df = file_df[train_end:]

    # build train
    train_df['TAXI_ID'] = train_df['TAXI_ID'].rank(method='dense').astype(int)
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(lambda s: sliding_window(json.loads(s), seq_len, window_len))
    train_df = train_df[train_df['POLYLINE'].map(len) > 0]
    train_df = train_df.explode('POLYLINE')
    # normalization
    tr_mean, tr_std = compute_mean_std(train_df)
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(lambda arr: (arr - tr_mean) / tr_std)
    train_df.to_pickle(train_file_path)

    # build val
    val_df['POLYLINE'] = val_df['POLYLINE'].transform(lambda s: json.loads(s)[:seq_len])
    val_df = val_df[val_df['POLYLINE'].map(len) > 0]
    val_df['POLYLINE_G'] = val_df['POLYLINE'].transform(lambda s: apply_gaussian_noise(s))
    val_df['POLYLINE_DS'] = val_df['POLYLINE'].transform(lambda s: apply_downsampling(s))
    val_df['POLYLINE'] = val_df['POLYLINE'].transform(lambda s: sliding_window(s, seq_len, window_len))
    val_df['POLYLINE_G'] = val_df['POLYLINE_G'].transform(lambda s: sliding_window(s, seq_len, window_len))
    val_df['POLYLINE_DS'] = val_df['POLYLINE_DS'].transform(lambda s: sliding_window(s, seq_len, window_len))
    val_df = val_df[val_df['POLYLINE'].map(len) > 0]
    val_df = val_df[val_df['POLYLINE_G'].map(len) > 0]
    val_df = val_df[val_df['POLYLINE_DS'].map(len) > 0]
    val_df['POLYLINE'] = val_df['POLYLINE'].transform(lambda arr: arr[0])
    val_df['POLYLINE_G'] = val_df['POLYLINE_G'].transform(lambda arr: arr[0])
    val_df['POLYLINE_DS'] = val_df['POLYLINE_DS'].transform(lambda arr: arr[0])

    # normalization
    val_df['POLYLINE'] = val_df['POLYLINE'].transform(lambda arr: (arr - tr_mean) / tr_std)
    val_df['POLYLINE_G'] = val_df['POLYLINE_G'].transform(lambda arr: (arr - tr_mean) / tr_std)
    val_df['POLYLINE_DS'] = val_df['POLYLINE_DS'].transform(lambda arr: (arr - tr_mean) / tr_std)
    val_df.to_pickle(val_file_path)


def sliding_window(lst: List, window_size: int, slide_step: int):
    output_len = 1 + max(0, int(math.ceil((len(lst) - window_size) / slide_step)))

    windowed_results = [
        calc_feature_stat_matrix(calc_car_movement_features(lst[step * slide_step:step * slide_step + window_size]))
        for step in range(output_len)
    ]

    return [element for element in windowed_results if element is not None]


def calc_car_movement_features(lst: List):
    if len(lst) < 4:
        return None

    gps_pos = np.array(lst, dtype=np.float64)
    seq_len, dim = gps_pos.shape
    result_seq_len = seq_len - 3

    velocity = gps_pos[1:] - gps_pos[:-1]
    velocity_norm = np.linalg.norm(gps_pos[1:] - gps_pos[:-1], axis=1)
    diff_velocity_norm = velocity_norm[1:] - velocity_norm[:-1]
    acceleration_norm = np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)
    diff_acceleration_norm = acceleration_norm[1:] - acceleration_norm[:-1]
    angular_velocity = np.arctan((gps_pos[1:, 1] - gps_pos[:-1, 1]) / (gps_pos[1:, 0] - gps_pos[:-1, 0] + EPSILON))

    return np.column_stack(
        (
            velocity_norm[:result_seq_len], diff_velocity_norm[:result_seq_len], acceleration_norm[:result_seq_len],
            diff_acceleration_norm[:result_seq_len], angular_velocity[:result_seq_len]
        )
    )


def calc_feature_stat_matrix(arr: np.ndarray):
    if arr is None:
        return None

    if arr.shape[0] == 0:
        return None

    pad_width = max(int(math.ceil(np.prod(arr.shape) / (2 * 5))) * 2, 4)
    arr = np.pad(arr, ((0, pad_width - arr.shape[0]), (0, 0)), 'symmetric')
    x = as_strided(arr, shape=(pad_width // 2 - 1, 4, 5), strides=(80, 40, 8))

    return np.column_stack(
        (
            x.mean(axis=1),
            x.min(axis=1, initial=FLOAT_MAX),
            x.max(axis=1, initial=FLOAT_MIN),
            x.std(axis=1),
            np.percentile(x, 25, axis=1),
            np.percentile(x, 50, axis=1),
            np.percentile(x, 75, axis=1),
        )
    )


def compute_mean_std(df: pd.DataFrame):
    _mean = df['POLYLINE'].transform(lambda arr: np.mean(arr, 0, keepdims=True)).to_numpy().mean()
    _std = df['POLYLINE'].transform(lambda arr: np.mean(arr, 0, keepdims=True)).to_numpy().std()
    # df['POLYLINE'] = df['POLYLINE'].transform(lambda arr: (arr - _mean) / _std)
    return _mean, _std


def normalize_local_metrics(df: pd.DataFrame):
    df['POLYLINE'] = df['POLYLINE'].transform(lambda arr: (arr - np.mean(arr, 0)) / np.std(arr, 0))


def apply_gaussian_noise(lst: List):
    trip = distort_gps_array(lst, .22, 50)
    trip[0] = lst[0]
    trip[-1] = lst[-1]
    return trip


def apply_downsampling(lst: List):
    return downsample_gps_array(lst, rate=DOWNSAMPLING_RATES[np.random.randint(0, 4 + 1)])


if __name__ == '__main__':
    import ray

    ray.shutdown()
    ray.init()

    prepare_taxi_data()
