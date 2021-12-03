"""

Prepare trajectory2vec data

script version: v4

"""

import json
import math
import random
from typing import List, Union

import modin.pandas as pd
import numpy as np

from preprocessing.common import DATASET_SAMPLE_RATE, downsampling_distort, EPSILON, FLOAT_MAX, FLOAT_MIN, \
    get_dataset_file, get_query_file, get_subtrajectories, get_trajectories_file, panda_types, save_pickle
from utils.array import downsampling
from utils.gps import distort, lonlat2meters


def prepare_taxi_data(in_file: str, out_prefix: str, seq_len=600, window_len=300, show_timestamps=True):
    train_prefix = out_prefix + ".train"
    val_prefix = out_prefix + ".val"
    test_prefix = out_prefix + ".test"

    file_df = pd.read_csv(in_file, dtype=panda_types, usecols=['POLYLINE'])

    file_df['POLYLINE'] = file_df['POLYLINE'].apply(lambda s: json.loads(s))
    file_df = file_df[file_df['POLYLINE'].map(len) >= 20]
    file_df = file_df[file_df['POLYLINE'].map(len) <= 100]
    file_df['POLYLINE'] = file_df['POLYLINE'].apply(lambda gps_list: gps2meters(gps_list))

    print("Cleaning Dataset")

    # shuffle and split
    train_size = 800_000
    test_size = 10_000
    val_size = 10_000
    train_df = file_df[:train_size]
    test_df = file_df[train_size:]

    print("Processing Train")

    # create source data
    train_df['SOURCE'] = train_df['POLYLINE'].transform(lambda gps_meter_list: downsampling_distort(gps_meter_list))

    # process target data
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(
        lambda gps_meter_list: sliding_window(gps_meter_list, seq_len, window_len, show_timestamps=show_timestamps)
    )
    # normalization
    tr_min, tr_max = compute_min_max(train_df)
    tr_diff = tr_max - tr_min
    train_df['POLYLINE'] = train_df['POLYLINE'].transform(lambda arr: (arr - tr_min) / tr_diff)

    # process source data
    train_df['SOURCE'] = train_df['SOURCE'].explode()
    print("exploded 'source' dataset")
    train_df['SOURCE'] = train_df['SOURCE'].apply(
        lambda gps_meter_list:
        sliding_window_varying_samplerate(gps_meter_list, seq_len, window_len, show_timestamps=show_timestamps)
    )
    train_df = train_df[train_df['SOURCE'].map(len) != 0]
    train_df['SOURCE'] = train_df['SOURCE'].transform(lambda arr: (arr - tr_min) / tr_diff)
    save_pickle(train_df, get_dataset_file(train_prefix))

    print("Generate: Validation")
    val_df = test_df.sample(n=val_size)
    test_df = test_df.loc[~test_df.index.isin(val_df.index)]
    # validation prepared exactly like training data

    val_df['POLYLINE'] = build_behavior_matrix(
        val_df['POLYLINE'], seq_len, window_len, tr_min, tr_diff, show_timestamps=show_timestamps
    )
    save_pickle(val_df['POLYLINE'], get_dataset_file(val_prefix))

    # experiment queries
    print("Experiment: TRAJECTORY SIMILARITY")
    # q - query trajectories
    # p - additional traj from the test set
    qp = test_df.sample(n=100_000 + test_size)
    q = qp[:test_size]
    p = qp[test_size:]

    q_a, q_b = get_subtrajectories(q['POLYLINE'])
    p_a, p_b = get_subtrajectories(p['POLYLINE'])
    query_db = pd.concat([q_b, p_a])  # p_b is ignored as we need only 100_000 traj

    save_pickle(q_a, get_trajectories_file(test_prefix, suffix="query"))
    save_pickle(query_db, get_trajectories_file(test_prefix, suffix="db"))

    q_a = build_behavior_matrix(q_a, seq_len, window_len, tr_min, tr_diff, show_timestamps=show_timestamps)
    save_pickle(q_a, get_query_file(test_prefix))

    query_db = build_behavior_matrix(query_db, seq_len, window_len, tr_min, tr_diff, show_timestamps=show_timestamps)
    save_pickle(query_db, get_dataset_file(test_prefix))

    d_sim = test_df.sample(n=test_size)
    save_pickle(d_sim['POLYLINE'], get_trajectories_file(test_prefix, suffix="similarity"))

    d_sim_processed = build_behavior_matrix(
        d_sim['POLYLINE'], seq_len, window_len, tr_min, tr_diff, show_timestamps=show_timestamps
    )
    save_pickle(d_sim_processed, get_dataset_file(test_prefix, suffix="similarity-ds_0.0"))

    for rate in [0.2, 0.4, 0.6]:
        print(f"downsampling rate : {rate}")
        downsampled_d_sim = d_sim['POLYLINE'].apply(lambda trip: downsampling(trip, rate))
        downsampled_d_sim = downsampled_d_sim.apply(
            lambda gps_meter_list:
            sliding_window_varying_samplerate(gps_meter_list, seq_len, window_len, show_timestamps=show_timestamps)
        )
        downsampled_d_sim = downsampled_d_sim.apply(lambda arr: (arr - tr_min) / tr_diff)

        save_pickle(downsampled_d_sim, get_dataset_file(test_prefix, suffix=f"similarity-ds_{rate}"))

    print("Experiment: DESTINATION PREDICTION")
    destination_task_trajectories = test_df.sample(n=test_size)

    destinations = destination_task_trajectories['POLYLINE'].apply(lambda trip: trip[-1])
    save_pickle(destinations, get_dataset_file(test_prefix, suffix="destinations"))

    destination_task_trajectories['POLYLINE'] = destination_task_trajectories['POLYLINE'].apply(
        lambda trip: trip[:int(len(trip) * 0.8)]
    )
    trajectory_queries = build_behavior_matrix(
        destination_task_trajectories['POLYLINE'],
        seq_len,
        window_len,
        tr_min,
        tr_diff,
        show_timestamps=show_timestamps
    )
    save_pickle(trajectory_queries, get_dataset_file(test_prefix, suffix="dp-traj-ds_0.0"))

    for rate in [0.2, 0.4, 0.6]:
        print(f"downsampling rate : {rate}")
        downsampled_d_dp = destination_task_trajectories['POLYLINE'].apply(lambda trip: downsampling(trip, rate))
        downsampled_d_dp = downsampled_d_dp.apply(
            lambda gps_meter_list:
            sliding_window_varying_samplerate(gps_meter_list, seq_len, window_len, show_timestamps=show_timestamps)
        )
        downsampled_d_dp = downsampled_d_dp.apply(lambda arr: (arr - tr_min) / tr_diff)

        save_pickle(downsampled_d_dp, get_dataset_file(test_prefix, suffix=f"dp-traj-ds_{rate}"))

    print("Experiment: TRAVEL-TIME ESTIMATION")
    traveltime_task_data = test_df.sample(n=test_size)
    travel_durations = traveltime_task_data['POLYLINE'].apply(lambda trip: len(trip) * 15)
    save_pickle(travel_durations, get_dataset_file(test_prefix, suffix="duration"))

    queries = build_behavior_matrix(
        traveltime_task_data['POLYLINE'],
        seq_len,
        window_len,
        tr_min,
        tr_diff,
        window_fn=sliding_window,
        show_timestamps=False
    )
    save_pickle(queries, get_dataset_file(test_prefix, suffix="tte-ds_0.0"))

    for rate in [0.2, 0.4, 0.6]:
        print(f"downsampling rate : {rate}")
        downsampled_d_tte = traveltime_task_data['POLYLINE'].apply(lambda trip: downsampling(trip, rate))
        downsampled_d_tte = downsampled_d_tte.apply(
            lambda gps_meter_list:
            sliding_window_varying_samplerate(gps_meter_list, seq_len, window_len, show_timestamps=show_timestamps)
        )
        downsampled_queries = downsampled_d_tte.apply(lambda arr: (arr - tr_min) / tr_diff)

        save_pickle(downsampled_queries, get_dataset_file(test_prefix, suffix=f"tte-ds_{rate}"))


def generate_input_trajectories(original_trips, train_df):
    print("Generating Input Sequences")
    temp_df = None
    drop_rates = [0.0, 0.2, 0.4, 0.6]
    distortion_rates = [0.0, 0.3, 0.6]
    for drop_rate in drop_rates:
        for distortion_rate in distortion_rates:
            print(f"Drop Rate: {drop_rate}; Distortion Rate: {distortion_rate}")
            cloned_df = train_df.copy(deep=False)
            cloned_df['SOURCE'] = original_trips.apply(
                lambda trip: distort(downsampling(trip, drop_rate), distortion_rate)
            )
            if temp_df is None:
                temp_df = cloned_df
            else:
                temp_df = pd.concat([temp_df, cloned_df])
    train_df = temp_df
    return train_df


def gps2meters(polyline: List):
    if len(polyline) == 0 or polyline is None:
        return None

    arr = np.array(polyline, dtype=np.float32)
    x, y = lonlat2meters(arr[:, 0], arr[:, 1])
    timesteps = np.arange(len(polyline)) * DATASET_SAMPLE_RATE
    return np.vstack((x, y, timesteps)).T


def sliding_window(arr: np.array, window_size_seconds: int, slide_step_seconds: int, show_timestamps=True):
    if arr is None:
        return []

    actual_sample_rate = int(arr[1, 2] - arr[0, 2])
    window_size = window_size_seconds // actual_sample_rate
    slide_step = slide_step_seconds // actual_sample_rate
    output_len = 1 + max(0, int(math.ceil((arr.shape[0] - window_size) / slide_step)))
    """
    make sure windowing does not need padding

    """
    movement_features, _ = calc_car_movement_features(arr, show_timestamps=show_timestamps)

    if movement_features is None:
        return []

    windowed_results = [
        calc_feature_stat_matrix(movement_features[step * slide_step:step * slide_step + window_size])
        for step in range(output_len)
    ]

    return [element for element in windowed_results if element is not None]


def sliding_window_varying_samplerate(
    arr: np.array, window_size_seconds: int, slide_step_seconds: int, show_timestamps=True
):
    movement_features, timesteps = calc_car_movement_features(arr, show_timestamps=show_timestamps)

    if movement_features is None:
        return []

    windows = rolling_window(movement_features, timesteps, window_size_seconds, slide_step_seconds)

    windowed_results = [calc_feature_stat_matrix(np.array(window)) for window in windows]

    return [element for element in windowed_results if element is not None]


def build_behavior_matrix(
    df: pd.Series,
    seq_len: int,
    window_len: int,
    tr_min: np.array,
    tr_diff: np.array,
    window_fn=sliding_window,
    show_timestamps=True
):
    df = df.apply(
        lambda gps_meter_list: window_fn(gps_meter_list, seq_len, window_len, show_timestamps=show_timestamps)
    )
    df = df.apply(lambda arr: (arr - tr_min) / tr_diff)
    return df


def rolling_window(sample, timesteps, window_size, offset):
    time_length = timesteps[len(sample) - 1]
    window_length = int(time_length / offset) + 1
    windows = [[] for _ in range(window_length)]

    for time, record in zip(timesteps, sample):
        for i in range(window_length):
            if (time > (i * offset)) & (time < (i * offset + window_size)):
                windows[i].append(record)
    return windows


def calc_car_movement_features(arr: np.array, show_timestamps=True) -> (Union[np.array, None], Union[np.array, None]):
    if arr.shape[0] <= 2:
        return None, None

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
            timesteps if show_timestamps else np.zeros_like(timesteps),
            np.pad(velocity_norm, (seq_len - velocity_norm.shape[0], 0), 'constant'),
            np.pad(diff_velocity_norm, (seq_len - diff_velocity_norm.shape[0], 0), 'constant'),
            np.pad(acceleration_norm, (seq_len - acceleration_norm.shape[0], 0), 'constant'),
            np.pad(diff_acceleration_norm, (seq_len - diff_acceleration_norm.shape[0], 0), 'constant'),
            np.pad(angular_velocity, (seq_len - angular_velocity.shape[0], 0), 'constant'),
        )
    ), timesteps


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


def compute_min_max(df: pd.DataFrame):
    _min = df['POLYLINE'].apply(lambda arr: np.min(arr, 0, initial=FLOAT_MAX)).to_list()
    _max = df['POLYLINE'].apply(lambda arr: np.max(arr, 0, initial=FLOAT_MIN)).to_list()

    _min = np.min(_min, axis=0)
    _max = np.max(_max, axis=0)
    return _min, _max


if __name__ == '__main__':
    random.seed(49)
    np.random.seed(49)

    import ray

    ray.shutdown()
    ray.init()

    prepare_taxi_data(
        in_file="../data/train.csv",
        out_prefix="../data/trajectory2vec-show_timestamps_3/trajectory2vec",
        seq_len=300,
        window_len=150,
        show_timestamps=True
    )
