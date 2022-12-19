# Import libs
import torch
import numpy as np


def get_attn_mask(summary_tokens_shape: int, 
                  device: str) -> torch.tensor:
    """Return a 2D mask in shape [summary_tokens_shape, summary_tokens_shape]

    Args:
        summary_tokens_shape (_type_): Number of tokens (Length) in the 
        summaries.
        device (_type_): The device in which the output tensor will be put on.

    Returns:
        torch.tensor: A torch.tensor mask.
    """
    attn_mask = np.triu(
        m=np.ones((summary_tokens_shape,
                   summary_tokens_shape), dtype=bool),
        k=1
    )
    return torch.tensor(attn_mask).to(device)
