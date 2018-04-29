import math
import torch.nn.functional as F

def my_metric(pred, target):
    assert len(pred) == len(target)
    return math.exp(F.nll_loss(F.log_softmax(pred, dim=1), target) / \
                    len(pred))