# Import libs
import torch
from torch import nn

class FnetMLPCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self,
                doc_tokens: torch.tensor,
                doc_token_types: torch.tensor,
                sum_tokens: torch.tensor,
                sum_token_types: torch.tensor):
        pass