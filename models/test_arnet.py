import unittest

import torch

from models.arnet import ARNet


class ReluTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ARNet(input_size=35, hidden_size=128, num_classes=5)

    def testValidInput(self):
        batch_sz = 5
        s = 128
        f = 35
        inp = torch.randn(batch_sz, s, f)
        self.model(inp)


if __name__ == '__main__':
    unittest.main(verbosity=2)
