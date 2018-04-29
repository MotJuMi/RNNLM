import torch.nn.functional as F

def my_loss(pred, target):
    return F.nll_loss(F.log_softmax(pred, dim=1), target)