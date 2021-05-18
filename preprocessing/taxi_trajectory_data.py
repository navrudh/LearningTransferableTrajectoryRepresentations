import json
import math
import multiprocessing
from typing import Callable, List

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from numpy.lib.stride_tricks import sliding_window_view
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
    train_df = pd.read_csv("../data/train.csv", dtype=panda_types, usecols=['TRIP_ID', 'TAXI_ID', 'POLYLINE'])
    train_df = applyParallel(train_df.groupby([]))
    train_df['POLYLINE_parsed'] = train_df['POLYLINE'].apply(lambda s: sliding_window_view(np.array(json.loads(s))))


def sliding_window(lst: List, window_size: int, slide_step: int):
    output_len = 1 + max(0, int(math.ceil((len(lst) - window_size) / slide_step)))
    return [lst[step * slide_step:step * slide_step + window_size] for step in range(output_len)]


def calc_car_movement_features(lst: List):
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


def applyParallel(dfGrouped, func: Callable):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)
