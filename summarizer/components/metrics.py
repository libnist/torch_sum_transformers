# Import libs
import torch
from torch import nn

# Replication of SparseCategoricalCrossEnropy in keras, wrapper

log_softmax = lambda x: torch.log(torch.softmax(x, dim=-1))
one_hot = lambda x, nc: nn.functional.one_hot(x, nc)

def NLLLossWrapper(fn):
    def inner(y_pred, y_true):
        mask = y_true[0] != 0
        loss_ = fn(log_softmax(y_pred[0]),
                  y_true[0])
        loss_ *= mask
        loss = torch.sum(loss_) / torch.sum(mask)
        batch_size = y_pred.shape[0]
        for i in range(1, batch_size):
            mask = y_true[i] != 0
            loss_ = fn(log_softmax(y_pred[i]),
                       y_true[i])
            loss_ *= mask
            loss += torch.sum(loss_) / torch.sum(mask)
        return loss / batch_size
    return inner

def CELossWrapper(fn, nc):
    def inner(y_pred, y_true):
        loss = fn(y_pred[0], one_hot(y_true[0], nc))
        batch_size = y_pred.shape[0]
        for i in range(1, batch_size):
            loss += fn(y_pred[i], one_hot(y_true[i], nc))
        return loss / batch_size
    return inner
