from typing import List, Tuple

import numpy as np


def downsample_gps_array(trip: List[Tuple[float]], rate: float):
    ds_trip = [trip[0]]
    for i in range(1, len(trip) - 1):
        if np.random.rand() > rate:
            ds_trip.append(trip[i])
    ds_trip.append(trip[-1])
    return ds_trip


def distort_gps_array(trip: List[Tuple[float]], rate: float, radius=50.0):
    noisetrip = trip.copy()
    for i, coords in enumerate(noisetrip):
        if np.random.rand() <= rate:
            x, y = lonlat2meters(*coords)
            xnoise, ynoise = 2 * np.random.rand() - 1, 2 * np.random.rand() - 1
            normz = np.hypot(xnoise, ynoise)
            xnoise, ynoise = xnoise * radius / normz, ynoise * radius / normz
            noisetrip[i] = meters2lonlat(x + xnoise, y + ynoise)
    return noisetrip


def lonlat2meters(lon: np.array, lat: np.array) -> Tuple[np.array, np.array]:
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = np.sin(north)
    return semimajoraxis * east, 3189068.5 * np.log((1 + t) / (1 - t))


def meters2lonlat(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = np.exp(y / 3189068.5)
    lat = np.arcsin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat
