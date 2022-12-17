# Import libs
import torch
from torch import nn

nllloss = nn.NLLLoss()

# Cross entropy loss


def nll_loss(y_pred: torch.tensor,
             y_true: torch.tensor):
    
    loss = nllloss(torch.log(torch.softmax(y_pred[0], dim=-1)),
                   y_true[0])
    for i in range(1, y_pred.shape[0]):
        loss += nllloss(torch.log(torch.softmax(y_pred[i], dim=-1)),
                        y_true[i])
    return loss/y_pred.shape[0]
