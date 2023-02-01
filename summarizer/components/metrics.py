# Import libs
import torch
from torch import nn

from types import FunctionType

import evaluate

# Replication of SparseCategoricalCrossEnropy in keras, wrapper


def log_softmax(x): return torch.log_softmax(x, dim=-1)
def one_hot(x, nc): return nn.functional.one_hot(x, nc)


def NLLLossWrapper(fn):
    def inner(y_pred, y_true):
        batch_size = y_pred.shape[0]
        loss = torch.tensor(0.0, requires_grad=True).to(y_pred.device)
        for i in range(batch_size):
            loss_ = fn(log_softmax(y_pred[i]),
                       y_true[i])
            loss = loss + loss_
        return loss / (batch_size)
    return inner


def CELossWrapper(fn: FunctionType, nc: int):
    def inner(y_pred, y_true):
        loss = fn(y_pred[0], one_hot(y_true[0], nc))
        batch_size = y_pred.shape[0]
        for i in range(1, batch_size):
            loss += fn(y_pred[i], one_hot(y_true[i], nc))
        return loss / batch_size
    return inner

def Accuracy(ignore_index=None):

    def accuracy(y_pred: torch.tensor,
                 y_true: torch.tensor):
        match_ = y_pred == y_true
        
        mask = y_true != ignore_index if ignore_index else y_true        

        match_ = match_ & mask
        return torch.sum(match_) / torch.sum(mask)
    return accuracy

rouge = evaluate.load("rouge")
