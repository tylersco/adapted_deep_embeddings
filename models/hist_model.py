'''
Adapted from https://danijar.com/structuring-your-tensorflow-models/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import os
import sys
import tensorflow as tf

from losses.histogram_loss import hist_loss
from .model import Model
from .utils import assign_to_device, _conv, define_scope, _fully_connected, get_available_gpus, _max_pooling, _relu, _softmax

class HistModel(Model):

    def __init__(self, config):
        self.config = self.get_config(config)
        self.saver = None
        self.learning_rate = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)

    def create_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def save_model(self, sess, step):
        self.saver.save(sess, os.path.join(self.config.save_dir_by_rep, 'model.ckpt'), global_step=step)

    def restore_model(self, sess):
        checkpoint = tf.train.latest_checkpoint(self.config.save_dir_by_rep)
        if checkpoint is None:
            sys.exit('Cannot restore model that does not exist')
        self.saver.restore(sess, checkpoint)

    def get_single_device(self):
        devices = get_available_gpus()
        d = self.config.controller
        if devices:
            d = devices[0]
        return d

    @define_scope
    def optimize(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            # Histogram loss placeholders
            self.pos_comps = tf.placeholder(tf.int32, [None, 2])
            self.n_pos_comps = tf.placeholder(tf.int32, ())
            self.neg_comps = tf.placeholder(tf.int32, [None, 2])
            self.n_neg_comps = tf.placeholder(tf.int32, ())

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            z = self.prediction

            loss = hist_loss(z, self.pos_comps, self.n_pos_comps, self.neg_comps, self.n_neg_comps)
            loss_reduce = tf.reduce_mean(loss)

            # NOTE: This model should be using batch norm, but Tensorflow hangs when trying to control
            # dependencies in newer versions (https://github.com/tensorflow/tensorflow/issues/19568).
            # The results from the paper used v1.4.0 before the bug was introduced.
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

            return train_op, loss_reduce, z

class MNISTHistModel(HistModel):

    def __init__(self, config):
        super().__init__(config)
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = tf.reshape(x, [-1, 28, 28, 1])
            x = _relu(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1))
            x = _max_pooling('pool2', _relu(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
            x = tf.contrib.layers.flatten(x)
            x = _fully_connected('fc1', x, 128)
            x = tf.nn.l2_normalize(x, axis=1)
            return x

class IsoletHistModel(HistModel):

    def __init__(self, config):
        super().__init__(config)
        self.input = tf.placeholder(tf.float32, [None, 617])
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = _relu(_fully_connected('fc1', x, 128))
            x = _fully_connected('fc2', x, 64)
            x = tf.nn.l2_normalize(x, axis=1)
            return x

class OmniglotHistModel(HistModel):

    def __init__(self, config):
        super().__init__(config)
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.9, epsilon=1e-5, fused=True, center=True, scale=False)
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = tf.reshape(x, [-1, 28, 28, 1])
            # NOTE: This model should be using batch norm, but Tensorflow hangs when trying to control
            # dependencies in newer versions (https://github.com/tensorflow/tensorflow/issues/19568).
            # The results from the paper used v1.4.0 before the bug was introduced.
            #x = _max_pooling('pool1', _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            #x = _max_pooling('pool2', _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _max_pooling('pool1', _relu(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
            x = _max_pooling('pool2', _relu(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
            #x = _relu(self.batch_norm(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train))
            x = _relu(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1))
            x = tf.contrib.layers.flatten(x)
            x = _fully_connected('fc1', x, 128)
            x = tf.nn.l2_normalize(x, axis=1)
            return x

class TinyImageNetHistModel(HistModel):

    def __init__(self, config):
        super().__init__(config)
        self.input = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.9, epsilon=1e-5, fused=True, center=True, scale=False)
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            # NOTE: This model should be using batch norm, but Tensorflow hangs when trying to control
            # dependencies in newer versions (https://github.com/tensorflow/tensorflow/issues/19568).
            # The results from the paper used v1.4.0 before the bug was introduced.
            #x = _max_pooling('pool1', _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            #x = _max_pooling('pool2', _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            #x = _max_pooling('pool3', _relu(self.batch_norm(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            #x = _relu(self.batch_norm(_conv('conv4', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train))
            x = _max_pooling('pool1', _relu(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
            x = _max_pooling('pool2', _relu(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
            x = _max_pooling('pool3', _relu(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
            x = _relu(_conv('conv4', x, 3, x.get_shape()[-1], 32, 1))
            x = tf.contrib.layers.flatten(x)
            x = _fully_connected('fc1', x, 128)
            x = tf.nn.l2_normalize(x, axis=1)
            return x
