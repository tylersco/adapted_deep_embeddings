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

from losses.proto_loss import prototypical_networks_loss
from .model import Model
from .utils import assign_to_device, _conv, define_scope, _fully_connected, get_available_gpus, _max_pooling, _relu, _softmax

class ProtoModel(Model):

    def __init__(self, support, query, label, config):
        self.support = support
        self.query = query
        self.label = label
        self.rows, self.cols = tf.shape(self.support)[-3], tf.shape(self.support)[-2]
        self.channels = tf.shape(self.support)[-1]
        self.support_reshape = tf.reshape(self.support, shape=[tf.shape(self.support)[0] * tf.shape(self.support)[1], self.rows, self.cols, self.channels])
        self.query_reshape = tf.reshape(self.query, shape=[tf.shape(self.query)[0] * tf.shape(self.query)[1], self.rows, self.cols, self.channels])
        self.input = tf.concat([support_reshape, query_reshape], 0)
        self.config = self.get_config(config)
        self.saver = None
        self.learning_rate = tf.placeholder(tf.float32)

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
            z = self.prediction
            z_dim = tf.shape(z)[-1]
            support, query = tf.split(z, [tf.shape(self.support_reshape)[0], tf.shape(self.query_reshape)[0]])
            z_proto = tf.reduce_mean(tf.reshape(support, shape=[tf.shape(self.support)[0], tf.shape(self.support)[1], z_dim]), axis=1)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            
            loss, accuracy, num_correct = prototypical_networks_loss(z_proto, query, tf.shape(self.query)[1], self.label)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)

        return train_op, loss, accuracy

class MNISTProtoModel(ProtoModel):

    def __init__(self, support, query, label, config):
        super().__init__(support, query, label, config)
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        x = self.input
        x = _relu(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1))
        x = _max_pooling('pool2', _relu(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = _fully_connected('fc1', x, 128)
        return x

class IsoletProtoModel(ProtoModel):

    def __init__(self, support, query, label, config):
        super().__init__(support, query, label, config)
        self.prediction
        self.optimize

    @define_scope
    def prediction(self):
        x = self.input
        x = _relu(_fully_connected('fc1', x, 128))
        x = _fully_connected('fc2', x, 64)
        return x

class OmniglotProtoModel(ProtoModel):

    def __init__(self, support, query, label, config):
        super().__init__(support, query, label, config)
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
        x = _fully_connected('fc1', x, 128)
        return x

class TinyImageNetProtoModel(ProtoModel):

    def __init__(self, support, query, label, config):
        super().__init__(support, query, label, config)
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
        x = _fully_connected('fc1', x, 128)
        return x
