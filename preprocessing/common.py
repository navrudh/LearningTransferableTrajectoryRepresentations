from pathlib import Path
from typing import Union

import modin.pandas as pd
import numpy as np

from utils.array import downsampling, downsampling_safe
from utils.gps import distort

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


def get_dataset_file(path, suffix=None):
    if suffix is not None:
        filename = "-".join([path, suffix])
    else:
        filename = path
    path = Path(f"{filename}.dataframe.pkl")
    print(f"Dataset File: {path}")
    return path


def get_trajectories_file(path, suffix=None):
    if suffix is not None:
        filename = "-".join([path, suffix])
    else:
        filename = path
    path = Path(f"{filename}.trajectories.pkl")
    print(f"Trajectories File: {path}")
    return path


def get_database_file(path):
    path = Path(f"{path}.query_database.pkl")
    print(f"Database File: {path}")
    return path


def get_query_file(path):
    path = Path(f"{path}.query.pkl")
    print(f"Query File: {path}")
    return path


def save_pickle(series: pd.Series, outfile: Union[str, Path]):
    series.to_pickle(
        str(outfile),
        compression="gzip",
    )

    print("Saved: " + str(outfile))


def get_subtrajectories(df: pd.DataFrame):
    df_a = df.apply(lambda polyline_list: polyline_list[::2])
    df_b = df.apply(lambda polyline_list: polyline_list[1::2])
    return df_a, df_b


def downsampling_distort(trip: np.ndarray, safe=False):
    noise_trips = []
    dropping_rates = [0, 0.2, 0.4, 0.6]
    distorting_rates = [0, 0.3, 0.6]
    for dropping_rate in dropping_rates:
        if safe:
            noisetrip1 = downsampling_safe(trip, dropping_rate)
        else:
            noisetrip1 = downsampling(trip, dropping_rate)
        for distorting_rate in distorting_rates:
            noisetrip2 = distort(noisetrip1, distorting_rate)
            noise_trips.append(noisetrip2)
    return noise_trips
