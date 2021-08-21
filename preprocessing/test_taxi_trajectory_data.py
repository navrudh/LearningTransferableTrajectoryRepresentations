import numpy as np
from numpy.lib.stride_tricks import as_strided

from preprocessing.taxi_trajectory_data import calc_feature_stat_matrix


def test_striding():
    arr = np.arange(0, 200).reshape(-1, 5)
    print(arr.shape)
    width = arr.shape[0]
    x = as_strided(arr, shape=(width // 2 - 1, 4, 5), strides=(80, 40, 8))
    print(x)

    y = calc_feature_stat_matrix(arr)
    print(y)


if __name__ == '__main__':
    test_striding()
