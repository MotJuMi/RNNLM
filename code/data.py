import os
import torch
from collections import defaultdict

class Vocabulary(object):
    def __init__(self):
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.UNK = self.w2i["<unk>"]
        self.EOS = self.w2i["<eos>"]
        self.i2w = defaultdict()

    def add_word(self, word, mode):
        if mode == "train":
            return self.w2i[word]

    def build_vocab(self):
        self.w2i = defaultdict(lambda: self.UNK, self.w2i)
        self.i2w = {v: k for k, v in self.w2i.items()}

    def __len__(self):
        return len(self.i2w)

class Corpus(object):
    def __init__(self, path="../data"):
        self.vocab = Vocabulary()
        self.train = self.get_data(os.path.join(path, "train.txt"))
        self.vocab.build_vocab()
        self.valid = self.get_data(os.path.join(path, "valid.txt"))
        self.test = self.get_data(os.path.join(path, "test.txt"))

    def get_data(self, path, mode, batch_size=10):
        assert os.path.exists(path)

        ids = torch.LongTensor()

        with open(path, 'r') as f:
            ids = []
            for line in f:
                ids.extend([self.vocab.add_word(x) for x in
                       line.strip().split(" ") + self.vocab.EOS])

        ids_tensor = torch.LongTensor(ids)
        ids_batched = self.batchify(ids_tensor, batch_size)

        return ids_batched

    def batchify(self, data, batch_size):
        num_batches = data.size(0) // batch_size
        data = data.narrow(0, 0, num_batches * batch_size)
        data = data.view(batch_size, -1).t().contiguous()

        return data





