import json
import math
import pickle
from pathlib import Path
from typing import List

import modin.pandas as pd
import numpy as np

from utils.gps import distort_gps_array, downsample_gps_array, lonlat2meters

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
SAMPLE_RATE = 15  # gps coordinated are sampled every 15 seconds


def get_dataset_file(path):
    return Path(f"{path}.dataframe.pkl")


def get_metadata_file(path):
    return Path(f"{path}.metadata.pkl")


def prepare_taxi_data(in_file: str, train_file_prefix: str, val_file_prefix: str, seq_len=600, window_len=300):
    if get_dataset_file(train_file_prefix).exists():
        print("Preprocessed taxi data file already exists")
        return
    file_df = pd.read_csv(in_file, dtype=panda_types, usecols=['POLYLINE'])

    metadata = {}
    # shuffle and split
    # file_df.reindex(np.random.permutation(file_df.index))
    # train_size = 0.9
    # train_end = int(len(file_df) * train_size)
    train_df = file_df
    # val_df = file_df[train_end:]

    # build train
    # train_df['TAXI_ID'] = train_df['TAXI_ID'].rank(method='dense').astype(int)
    # train_df['POLYLINE'] = train_df['POLYLINE'].transform(lambda s: gps2meters(json.loads(s)))
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(
        lambda s: sliding_window(gps2meters(json.loads(s)), seq_len, window_len)
    )
    train_df = train_df[train_df['POLYLINE'].map(len) > 0]
    # train_df = train_df.explode('POLYLINE')
    # normalization
    tr_min, tr_max = compute_min_max(train_df)
    tr_diff = tr_max - tr_min
    metadata["train_min"] = tr_min
    metadata["train_max"] = tr_max
    pickle.dump(metadata, open(get_metadata_file(train_file_prefix), "wb"))
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(lambda arr: (arr - tr_min) / tr_diff)
    train_df['POLYLINE'].to_pickle(get_dataset_file(train_file_prefix))

    # build val
    # val_df['POLYLINE'] = val_df['POLYLINE'].transform(lambda s: json.loads(s)[:seq_len])
    # val_df = val_df[val_df['POLYLINE'].map(len) > 0]
    # val_df['POLYLINE_G'] = val_df['POLYLINE'].transform(lambda s: apply_gaussian_noise(s))
    # val_df['POLYLINE_DS'] = val_df['POLYLINE'].transform(lambda s: apply_downsampling(s))
    # val_df['POLYLINE'] = val_df['POLYLINE'].transform(lambda s: sliding_window(s, seq_len, window_len))
    # val_df['POLYLINE_G'] = val_df['POLYLINE_G'].transform(lambda s: sliding_window(s, seq_len, window_len))
    # val_df['POLYLINE_DS'] = val_df['POLYLINE_DS'].transform(lambda s: sliding_window(s, seq_len, window_len))
    # val_df = val_df[val_df['POLYLINE'].map(len) > 0]
    # val_df = val_df[val_df['POLYLINE_G'].map(len) > 0]
    # val_df = val_df[val_df['POLYLINE_DS'].map(len) > 0]
    # val_df['POLYLINE'] = val_df['POLYLINE'].transform(lambda arr: arr[0])
    # val_df['POLYLINE_G'] = val_df['POLYLINE_G'].transform(lambda arr: arr[0])
    # val_df['POLYLINE_DS'] = val_df['POLYLINE_DS'].transform(lambda arr: arr[0])

    # normalization
    # val_df['POLYLINE'] = val_df['POLYLINE'].transform(lambda arr: (arr - tr_mean) / tr_std)
    # val_df['POLYLINE_G'] = val_df['POLYLINE_G'].transform(lambda arr: (arr - tr_mean) / tr_std)
    # val_df['POLYLINE_DS'] = val_df['POLYLINE_DS'].transform(lambda arr: (arr - tr_mean) / tr_std)
    # val_df.to_pickle(val_file_path)


def gps2meters(lst: List):
    arr = np.array(lst, dtype=np.float32)
    x, y = lonlat2meters(arr[:, 0], arr[:, 1])
    return np.vstack((x, y)).T


def sliding_window(arr: np.array, window_size_seconds: int, slide_step_seconds: int):
    window_size = window_size_seconds // SAMPLE_RATE
    slide_step = slide_step_seconds // SAMPLE_RATE
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


def calc_car_movement_features(arr: np.array):
    if arr.shape[0] < 4:
        return None

    gps_pos = arr
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


if __name__ == '__main__':
    import ray

    ray.shutdown()
    ray.init()

    prepare_taxi_data(
        in_file="../data/train.csv",
        train_file_prefix="../data/preprocessed-trajectory2vec-porto-train",
        val_file_prefix="../data/preprocessed-trajectory2vec-porto-val",
        seq_len=600,
        window_len=300
    )
