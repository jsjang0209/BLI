import torch
import torch.nn as nn


class CodingScheduler():
    def __init__(self):
        super().__init__()

        self.num_types = 3
        self.masks = {}
        self.count = 0
        self.L = 4

    def _count(self):
        self.count += 1
        if self.count == self.L:
            self.count = 0
