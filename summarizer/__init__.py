from .summarizer import GreedySummarizer

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_transformer_scheduler(optimizer: torch.optim.Optimizer,
                              warmup_steps: int,
                              d_model: int,
                              last_epoch: int = -1):

    warmup_coeff = warmup_steps**-1.5

    # Inverse of the optimizers default lr is used to neutrize the effect of it.
    d_model_coeff = (d_model**-0.5) * (1 / optimizer.param_groups[0]["lr"])

    def lr_lambda(current_step):
        current_step += 1
        return d_model_coeff * min(current_step**-0.5,
                                   current_step * warmup_coeff)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_modified_transformer_scheduler(optimizer: torch.optim.Optimizer,
                                       warmup_steps: int,
                                       coeff: float = 0.002,
                                       last_epoch: int = -1):

    warmup_coeff = warmup_steps**-1.5

    # Inverse of the optimizers default lr is used to neutrize the effect of it.
    coeff = coeff * (1 / optimizer.param_groups[0]["lr"])

    def lr_lambda(current_step):
        current_step += 1
        return coeff * min(current_step**-0.5, current_step * warmup_coeff)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
