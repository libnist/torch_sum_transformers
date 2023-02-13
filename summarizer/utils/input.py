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

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return torch.tensor(pos_encoding).unsqueeze(0)