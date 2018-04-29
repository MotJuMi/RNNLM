import logging
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires._grad,
                                  self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)