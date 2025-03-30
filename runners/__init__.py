#from .STFRunner import STFRunner
from .LTSFRunner import LTSFRunner

#__all__ = ['BaseRunner', 'STFRunner', 'LTSFRunner']
__all__ = ['BaseRunner', 'LTSFRunner']


def select_runner(name):

    # TODO: Implement STFRunner.
    if name == 'LTSFRunner':
        return LTSFRunner
    #elif name == 'STFRunner':
    #    return STFRunner        
    else:
        raise NotImplementedError()