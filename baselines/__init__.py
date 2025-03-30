from .DSTMamba import DSTMamba

def select_model(name):
    model_dict = {
        'DSTMamba': DSTMamba
    }

    return model_dict[name]