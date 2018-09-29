'''
Adapted from https://danijar.com/structuring-your-tensorflow-models/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from .utils import AttrDict

ABC = abc.ABCMeta('ABC', (object,), {})

class Model(ABC):

    def __init__(self):
        super().__init__()

    def get_config(self, config):
        c = AttrDict()
        for k, v in config.items():
            c[k] = v
        return c

    @abc.abstractmethod
    def prediction(self):
        pass

    @abc.abstractmethod
    def optimize(self):
        pass
