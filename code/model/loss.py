import torch.nn.functional as F

def my_loss(pred, target):
    #print('LOSS:')
    #print('pred.size(): ' + str(pred.size()))
    #print('target.size(): ' + str(target.size()))
    #log_sm = F.log_softmax(pred, dim=1)
    #print('log_sm.size(): ' + str(log_sm.size()))
    return F.nll_loss(F.log_softmax(pred, dim=1), target)