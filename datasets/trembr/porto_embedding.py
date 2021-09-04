import csv
import os
import random

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
"""dataset + accelerate library of Huggingface"""


class PortoIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, id_edge_raw_file):
        self.data = ["1,2,3,4,5", "5,6,7"]
        self.idx2type = {}
        self.segment_ids = []

        self._initialize_idx2type(os.path.expanduser(id_edge_raw_file))

    def _initialize_idx2type(self, file):
        type_dict = {}
        type_count = 0
        with open(file, "r") as id_edge_raw:
            reader = csv.reader(id_edge_raw, delimiter=";")
            for row in reader:
                segment_id, type = int(row[0]), row[6]
                # print(segment_id, type)
                if type not in type_dict:
                    type_dict[type] = type_count
                    type_count += 1
                self.idx2type[segment_id] = type_dict[type]
        self.segment_ids = list(self.idx2type.keys())

    def _negative_sample(self, segment1):
        segment2 = random.choice(self.segment_ids)
        return np.array([segment2, segment1]), 0, self._segment_types_match(segment1, segment2)

    def _segment_types_match(self, segment1, segment2):
        if segment1 not in self.idx2type or segment2 not in self.idx2type:
            return 0
        return self.idx2type[segment1] == self.idx2type[segment2]

    def parse_file(self):
        for line in self.data:
            segments = list(map(int, line.split(",")))
            for i in range(len(segments) - 1):
                for j in range(min(i + 4, len(segments))):
                    """
                    4 negative samples for every positive sample
                    """
                    yield np.array([segments[i], segments[j]]), 1, self._segment_types_match(segments[i], segments[j])
                    yield self._negative_sample(segments[i])
                    yield self._negative_sample(segments[j])
                    yield self._negative_sample(segments[j])
                    yield self._negative_sample(segments[i])

    def get_stream(self):
        return self.parse_file()

    def __iter__(self):
        return self.get_stream()


if __name__ == '__main__':
    iterable_dataset = PortoIterableDataset(id_edge_raw_file="~/Downloads/id_edge_raw.txt")
    loader = DataLoader(iterable_dataset, batch_size=8, drop_last=True)

    for batch in loader:
        print(batch)
