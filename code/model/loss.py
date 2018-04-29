import torch.nn.functional as F

def my_loss(pred, target):
    return F.nll_loss(pred, target)