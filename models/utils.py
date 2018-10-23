'''
Decorators adapted from https://danijar.com/structuring-your-tensorflow-models/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def doublewrap(function):
    '''
    A decorator around a decorator allowing use of the original decorator
    without parentheses if no arguments are provided. All arguments
    must be optional.
    '''
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    '''
    A decorator for functions that define Tensorflow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result, so that operations are only added to the graph once.
    '''
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]

def assign_to_device(device, ps_device):
    '''
    Returns a function to place variables on the ps_device.\
    If ps_device is not set, then the variables will be placed
    on the default device.
    '''
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign

def _stride(s):
    return [1, s, s, 1]

def _relu(x):
    return tf.nn.relu(x)

def _sigmoid(x):
    return tf.nn.sigmoid(x)

def _tanh(x):
    return tf.nn.tanh(x)

def _softmax(x):
    return tf.nn.softmax(x)

def _max_pooling(name, x, kernel_size, stride, padding='SAME'):
    return tf.nn.max_pool(x, ksize=_stride(kernel_size), strides=_stride(stride), padding=padding, name=name)

def _conv(name, x, filter_size, in_size, out_size, stride, padding='SAME', bias=True, reuse=None, weight_decay=None):
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable(
            'conv_weights', [filter_size, filter_size, in_size, out_size],
            tf.float32, initializer=tf.initializers.random_normal()
        )

        res = tf.nn.conv2d(x, weights, _stride(stride), padding=padding)

        if bias:
            biases = tf.get_variable(
                'conv_biases', [out_size], tf.float32,
                initializer=tf.initializers.random_normal()
            )
            res += biases

        if weight_decay:
            wd = tf.nn.l2_loss(weights) * weight_decay
            tf.add_to_collection('weight_decay', wd)

    return res

def _fully_connected(name, x, out_size, reuse=None, weight_decay=None):
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable(
            'fc_weights', [x.get_shape()[1], out_size], tf.float32,
            initializer=tf.initializers.random_normal()
        )
        biases = tf.get_variable(
            'fc_biases', [out_size], tf.float32, initializer=tf.initializers.random_normal()
        )

        if weight_decay:
            wd = tf.nn.l2_loss(weights) * weight_decay
            tf.add_to_collection('weight_decay', wd)

        return tf.nn.xw_plus_b(x, weights, biases)
