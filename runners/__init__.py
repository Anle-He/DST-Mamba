from .STFRunner import STFRunner
from .LTSFRunner import LTSFRunner

__all__ = ['BaseRunner', 'STFRunner', 'LTSFRunner']


def select_runner(name):

    if name == 'STFRunner':
        return STFRunner
    if name == 'LTSFRunner':
        return LTSFRunner      
    else:
        raise NotImplementedError()