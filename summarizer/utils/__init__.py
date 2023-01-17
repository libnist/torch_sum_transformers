from .data import get_data, get_dataloader, get_fnet_pretrained_dataloader
from .input import get_attn_mask

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

def summary_writer(path: str,
                   model_name: str,
                   time: str = None,
                   extra: str = None):
    if not time:
        time = datetime.now().strftime("%Y-%m-%d")
    
    if not extra:
        log_dir = os.path.joint(path, model_name, time)
    else:
        log_dir = os.path.joint(path, model_name, time, extra)
        
    return SummaryWriter(log_dir=log_dir)
        