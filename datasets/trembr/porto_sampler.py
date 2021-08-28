import collections

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

"""dataset + accelerate library of Huggingface"""


class PortoIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self):
        self.data = ["1,2,3,4,5", "5,6,7"]
        self.idx2type = collections.defaultdict(int)

    def parse_file(self):
        for line in self.data:
            vals = list(map(int, line.split(",")))
            for i in range(len(vals) - 1):
                for j in range(min(i + 4, len(vals))):
                    yield np.array([vals[i], vals[j]]), 1, self.idx2type[vals[i]] == self.idx2type[vals[j]]
                    yield np.array([np.random.randint(0, 108785), vals[j]]), 0, 0
                    yield np.array([np.random.randint(0, 108785), vals[j]]), 0, 0
                    yield np.array([np.random.randint(0, 108785), vals[i]]), 0, 0
                    yield np.array([np.random.randint(0, 108785), vals[i]]), 0, 0

    def get_stream(self):
        return self.parse_file()

    def __iter__(self):
        return self.get_stream()


if __name__ == '__main__':
    iterable_dataset = PortoIterableDataset()
    loader = DataLoader(iterable_dataset, batch_size=8, drop_last=True)

    for batch in loader:
        print(batch)
