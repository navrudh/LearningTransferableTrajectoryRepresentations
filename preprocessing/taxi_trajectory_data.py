import json
import math
from pathlib import Path
from typing import List

import modin.pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided

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


def prepare_taxi_data(seq_len=256, window_len=32):
    file_path = Path("../data/train-preprocessed-taxi.pkl")
    if file_path.exists():
        print("Preprocessed taxi data file already exists")
        return
    train_df = pd.read_csv("../data/train.csv", dtype=panda_types, usecols=['TAXI_ID', 'POLYLINE'])
    train_df['TAXI_ID'] = train_df['TAXI_ID'].rank(method='dense').astype(int)
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(lambda s: sliding_window(json.loads(s), seq_len, window_len))
    train_df = train_df[train_df['POLYLINE'].map(len) > 0]
    train_df = train_df.explode('POLYLINE')
    train_df.to_pickle(file_path)


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


def apply_sliding_window_to_polyline(df: pd.DataFrame):
    df['POLYLINE'].apply(lambda s: json.loads(s))
    return df


if __name__ == '__main__':
    import ray

    ray.shutdown()
    ray.init()

    prepare_taxi_data()
