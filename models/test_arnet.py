import random
import unittest

import torch
from torch.nn.utils.rnn import pack_sequence

from models.arnet import ARNet


class ArnetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ARNet(input_size=35, hidden_size=128, num_classes=5)

    def testValidInput(self):
        batch_sz = 5
        s = 128
        f = 35
        inp = torch.randn(batch_sz, s, f)
        self.model(inp)

    def testValidInput2(self):
        batch_sz = 5
        s = 128
        f = 35
        inp = pack_sequence([torch.randn(random.randint(1, s), f) for _ in range(batch_sz)], enforce_sorted=False)
        self.model(inp)


if __name__ == '__main__':
    unittest.main(verbosity=2)
