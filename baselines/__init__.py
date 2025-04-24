from .SMamba import SMamba
from .DSTMamba import DSTMamba

def select_model(name):
    model_dict = {
        'SMamba': SMamba,
        'DSTMamba': DSTMamba
    }

    return model_dict[name]