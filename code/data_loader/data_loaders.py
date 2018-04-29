import torch
import numpy as np
from torchtext import datasets
from base import BaseDataLoader

class WikiTextDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(WikiTextDataLoader, self).__init__(config)
        self.data_dir = config['data_loader']['data_dir']
        self.data_loader = torch.utils.data.DataLoader(
            datasets.WikiText2('../data', train=True, download=True),
                               batch_size=256, shuffle=False)
        self.x = []
        self.y = []
        for data, target in self.data_loader:
            self.x += [i for i in data.numpy()]
            self.y += [i for i in target.numpy()]
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __next__(self):
        batch = super(WikiTextDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]
        return batch

    def _pack_data(self):
        packed = list(zip(self.x, self.y))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.x, self.y = unpacked

    def _n_samples(self):
        return len(self.x)