# Import libs
import torch
from torch import nn
import torch.nn.functional as F

cross_entropy = nn.CrossEntropyLoss()

# Cross entropy loss
def cross_entropy_loss(y_pred: torch.tensor,
                       y_true: torch.tensor,
                       vocab_size: int):
    categorical_y_true = F.one_hot(y_true, vocab_size).float()
    return cross_entropy(y_pred, categorical_y_true)