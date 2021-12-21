import numpy as np


def downsampling(arr: np.array, rate: float):
    if rate == 0.0:
        return arr
    keep = np.random.random_sample(size=arr.shape[0])
    return arr[keep > rate]


def downsampling_safe(arr: np.array, rate: float):
    if rate == 0.0:
        return arr
    keep = np.random.random_sample(size=arr.shape[0])
    keep[0] = 1
    keep[-1] = 1
    return arr[keep > rate]


if __name__ == '__main__':
    print(downsampling(np.arange(20), 0.5))
