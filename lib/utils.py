import json
import numpy as np

import torch


def print_log(*values, log=None, end='\n'):

    print(*values, end=end)

    if log:
        if isinstance(log, str):
            log = open(log, 'a')
        print(*values, file=log, end=end)
        log.flush()


class CustomJSONEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f'Shape: {obj.shape}'
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)