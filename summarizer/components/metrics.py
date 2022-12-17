# Import libs
import torch
from torch import nn

nllloss = nn.NLLLoss()

# Replication of SparseCategoricalCrossEnropy in keras, wrapper

log_softmax = lambda x: torch.log(torch.softmax(x, dim=-1))

def NLLLossWrapper(fn):
    def inner(y_pred, y_true):
        loss = fn(log_softmax(y_pred[0]),
                  y_true[0])
        batch_size = y_pred.shape[0]
        for i in range(1, batch_size):
            loss += fn(log_softmax(y_pred[i]),
                       y_true[i])
        return loss / batch_size
    return inner
