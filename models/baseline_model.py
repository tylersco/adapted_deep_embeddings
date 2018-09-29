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

from .model import Model
from .utils import assign_to_device, _conv, define_scope, _fully_connected, get_available_gpus, _max_pooling, _relu, _softmax

class BaselineModel(Model):

    def __init__(self, input, target, config):
        self.input = input
        self.target = target
        self.config = self.get_config(config)
        self.saver = None
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

    def create_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def save_model(self, sess, step):
        self.saver.save(sess, os.path.join(self.config.save_dir, 'model.ckpt'), global_step=step)

    def restore_model(self, sess):
        checkpoint = tf.train.latest_checkpoint(self.config.save_dir)
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
            pred = self.prediction
            correct_pred = tf.equal(tf.argmax(pred, 1), self.target)
            num_correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, 
                labels=tf.one_hot(self.target, self.target.get_shape()[-1])))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(cost)

        return train_op, cost, num_correct, accuracy

class MNISTBaselineModel(BaselineModel):

    def __init__(self, input, target, config):
        super().__init__(input, target, config)
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        x = self.input
        x = _relu(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1))
        x = _max_pooling('pool2', _relu(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = _relu(_fully_connected('fc1', x, 128))
        x = _fully_connected('fc1', x, self.target.get_shape()[-1])
        return x

class IsoletBaselineModel(BaselineModel):

    def __init__(self, input, target, config):
        super().__init__(input, target, config)
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        x = self.input
        x = _relu(_fully_connected('fc1', x, 128))
        x = _relu(_fully_connected('fc2', x, 64))
        x = _fully_connected('fc3', x, self.target.get_shape()[-1])
        return x

class OmniglotBaselineModel(BaselineModel):

    def __init__(self, input, target, target2, config):
        super().__init__(input, target, config)
        self.is_train = tf.placeholder(tf.bool)
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.1, epsilon=1e-5, fused=True, center=True, scale=False)
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        x = self.input
        x = _max_pooling('pool1', _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
        x = _max_pooling('pool2', _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
        x = _relu(self.batch_norm(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train))
        x = tf.contrib.layers.flatten(x)
        x = _relu(_fully_connected('fc1', x, 128))
        x = _fully_connected('fc2', x, self.target.get_shape()[-1])
        return x

class TinyImageNetBaselineModel(BaselineModel):

    def __init__(self, input, target, target2, config):
        super().__init__(input, target, config)
        self.is_train = tf.placeholder(tf.bool)
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.1, epsilon=1e-5, fused=True, center=True, scale=False)
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        x = self.input
        x = _max_pooling('pool1', _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
        x = _max_pooling('pool2', _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
        x = _max_pooling('pool3', _relu(self.batch_norm(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
        x = _relu(self.batch_norm(_conv('conv4', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train))
        x = tf.contrib.layers.flatten(x)
        x = _relu(_fully_connected('fc1', x, 128))
        x = _fully_connected('fc2', x, self.target.get_shape()[-1])
        return x
