"""

Prepare trajectory2vec data

script version: v4

"""
import json
import os
import random

import modin.pandas as pd
import numpy as np

import pandas

from preprocessing.common import get_dataset_file, panda_types, save_pickle
from preprocessing.taxi_trajectory_data_trajectory2vec import compute_min_max, gps2meters, sliding_window


def prepare_taxi_data(in_file: str, out_prefix: str, norm_file: str, seq_len=600, window_len=300, show_timestamps=True,
                      ):
    test_prefix = out_prefix + ".test"

    print("Normalization: Compute Min, Max")
    norm_df = pd.read_csv(norm_file, dtype=panda_types, usecols=['POLYLINE'])
    print("Read " + str(len(norm_df)) + " records")

    norm_series = norm_df['POLYLINE'].apply(lambda s: json.loads(s))
    norm_series = norm_series.apply(lambda gps_list: gps2meters(gps_list))
    norm_series = norm_series.apply(
        lambda gps_meter_list: sliding_window(gps_meter_list, seq_len, window_len, show_timestamps=show_timestamps)
    )
    # normalization
    tr_min, tr_max = compute_min_max(norm_series)
    tr_diff = tr_max - tr_min

    print("Test Dataset: Create")
    file_df = pandas.read_csv(in_file, dtype=panda_types, usecols=['POLYLINE'])
    print("Read " + str(len(file_df)) + " records")

    test_series = file_df['POLYLINE'].apply(lambda s: json.loads(s))
    test_series = test_series.apply(lambda gps_list: gps2meters(gps_list))
    test_series = test_series.apply(
        lambda gps_meter_list: sliding_window(gps_meter_list, seq_len, window_len, show_timestamps=show_timestamps)
    )
    test_series = test_series.apply(lambda arr: (arr - tr_min) / tr_diff)

    save_pickle(test_series, get_dataset_file(test_prefix))


if __name__ == '__main__':
    random.seed(49)
    np.random.seed(49)

    import ray

    ray.shutdown()
    ray.init()

    output_dir = "../data/simulated/trajectory2vec"
    os.makedirs(output_dir, exist_ok=True)

    prepare_taxi_data(
        in_file="../data/simulated/test.csv",
        norm_file="../data/simulated/train.csv",
        out_prefix=f"{output_dir}/trajectory2vec-test",
        seq_len=300,
        window_len=150,
        show_timestamps=True
    )
