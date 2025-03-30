import os
import torch
import random
import numpy as np

from typing import Optional, NoReturn


def seed_everything(random_seed: Optional[int] = 42) -> NoReturn:
    
    if seed is None:
        return
        
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # If you are using multi-GPU.

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False