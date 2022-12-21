# Import libs
import torch
from torch import nn

from types import FunctionType

import evaluate

# Replication of SparseCategoricalCrossEnropy in keras, wrapper


def log_softmax(x): return torch.log(torch.softmax(x, dim=-1))
def one_hot(x, nc): return nn.functional.one_hot(x, nc)


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


def CELossWrapper(fn: FunctionType, nc: int):
    def inner(y_pred, y_true):
        loss = fn(y_pred[0], one_hot(y_true[0], nc))
        batch_size = y_pred.shape[0]
        for i in range(1, batch_size):
            loss += fn(y_pred[i], one_hot(y_true[i], nc))
        return loss / batch_size
    return inner

NllLoss = NLLLossWrapper(nn.NLLLoss(reduction="none"))

def accuracy(y_pred: torch.tensor,
             y_true: torch.tensor):
    """Calculates the accuracy of a summarization model.

    Args:
        y_pred (torch.tensor): Predicted tokens.
        y_true (torch.tensor): Ground truth tokens.

    Returns:
        _type_: _description_
    """
    match_ = y_pred == y_true
    mask = y_true != 0

    match_ = match_ & mask
    return torch.sum(match_) / torch.sum(mask)

rouge = evaluate.load("rouge")
