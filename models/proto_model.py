'''
Adapted from https://danijar.com/structuring-your-tensorflow-models/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import numpy as np
import os
import sys
import tensorflow as tf

from losses.proto_loss import prototypical_networks_loss
from .model import Model
from .utils import assign_to_device, _conv, define_scope, _fully_connected, get_available_gpus, _max_pooling, _relu, _softmax

class ProtoModel(Model):

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
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            loss, accuracy, num_correct = self.metrics
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)

            return train_op, loss, accuracy

    @define_scope(scope='stream_metrics')
    def metrics(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            z = self.prediction
            z_dim = tf.shape(z)[-1]
            support, query = tf.split(z, [tf.shape(self.support_reshape)[0], tf.shape(self.query_reshape)[0]])
            z_proto = tf.reduce_mean(tf.reshape(support, shape=[tf.shape(self.support)[0], tf.shape(self.support)[1], z_dim]), axis=1)
            loss, accuracy, num_correct = prototypical_networks_loss(z_proto, query, tf.shape(self.query)[1], self.label)
            return loss, accuracy, num_correct

class MNISTProtoModel(ProtoModel):

    def __init__(self, config):
        super().__init__(config)
        self.support = tf.placeholder(tf.float32, [None, None, 784])
        self.query = tf.placeholder(tf.float32, [None, None, 784])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.support_reshape = tf.reshape(self.support, shape=[tf.shape(self.support)[0] * tf.shape(self.support)[1], 28, 28, 1])
        self.query_reshape = tf.reshape(self.query, shape=[tf.shape(self.query)[0] * tf.shape(self.query)[1], 28, 28, 1])
        self.input = tf.concat([self.support_reshape, self.query_reshape], 0)
        self.prediction
        self.optimize
        self.metrics

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = _relu(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1))
            x = _max_pooling('pool2', _relu(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1)), 2, 2)
            x = tf.contrib.layers.flatten(x)
            x = _fully_connected('fc1', x, 128)
            return x

class IsoletProtoModel(ProtoModel):

    def __init__(self, config):
        super().__init__(config)
        self.support = tf.placeholder(tf.float32, [None, None, 617])
        self.query = tf.placeholder(tf.float32, [None, None, 617])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.support_reshape = tf.reshape(self.support, shape=[tf.shape(self.support)[0] * tf.shape(self.support)[1], 617])
        self.query_reshape = tf.reshape(self.query, shape=[tf.shape(self.query)[0] * tf.shape(self.query)[1], 617])
        self.input = tf.concat([self.support_reshape, self.query_reshape], 0)
        self.prediction
        self.optimize
        self.metrics

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = _relu(_fully_connected('fc1', x, 128))
            x = _fully_connected('fc2', x, 64)
            return x

class OmniglotProtoModel(ProtoModel):

    def __init__(self, config):
        super().__init__(config)
        self.support = tf.placeholder(tf.float32, [None, None, 784])
        self.query = tf.placeholder(tf.float32, [None, None, 784])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.support_reshape = tf.reshape(self.support, shape=[tf.shape(self.support)[0] * tf.shape(self.support)[1], 28, 28, 1])
        self.query_reshape = tf.reshape(self.query, shape=[tf.shape(self.query)[0] * tf.shape(self.query)[1], 28, 28, 1])
        self.input = tf.concat([self.support_reshape, self.query_reshape], 0)
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.1, epsilon=1e-5, fused=True, center=True, scale=False)
        self.prediction
        self.optimize
        self.metrics

    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.input
            x = _max_pooling('pool1', _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _max_pooling('pool2', _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _relu(self.batch_norm(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train))
            x = tf.contrib.layers.flatten(x)
            x = _fully_connected('fc1', x, 128)
            return x

class TinyImageNetProtoModel(ProtoModel):

    def __init__(self, config):
        super().__init__(config)
        self.p = tf.placeholder(tf.float32, [None, 128])
        self.query = tf.placeholder(tf.float32, [None, None, 64, 64, 3])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.query_reshape = tf.reshape(self.query, shape=[tf.shape(self.query)[0] * tf.shape(self.query)[1], 64, 64, 3])
        self.batch_norm = partial(tf.layers.batch_normalization,
            momentum=0.1, epsilon=1e-5, fused=True, center=True, scale=False)
        self.prediction
        self.optimize
        self.metrics

    def compute_batch_prototypes(self, sess, support_batch, total_classes, num_classes_per_batch=10):
        prototypes = np.zeros((total_classes, 128))
        for i in range(0, len(support_batch), num_classes_per_batch):
            s = support_batch[i:i + num_classes_per_batch]
            feed_dict = {
                self.query: s,
                self.is_train: False
            }
            z_p = sess.run(self.prediction_proto, feed_dict=feed_dict)
            prototypes[i:i + num_classes_per_batch] = z_p
        return prototypes
    
    @define_scope
    def prediction(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            x = self.query_reshape
            x = _max_pooling('pool1', _relu(self.batch_norm(_conv('conv1', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _max_pooling('pool2', _relu(self.batch_norm(_conv('conv2', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _max_pooling('pool3', _relu(self.batch_norm(_conv('conv3', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train)), 2, 2)
            x = _relu(self.batch_norm(_conv('conv4', x, 3, x.get_shape()[-1], 32, 1), training=self.is_train))
            x = tf.contrib.layers.flatten(x)
            x = _fully_connected('fc1', x, 128)
            return x

    @define_scope
    def prediction_proto(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            proto = self.prediction
            return tf.reduce_mean(tf.reshape(proto, [tf.shape(self.query)[0], tf.shape(self.query)[1], tf.shape(proto)[-1]]), axis=1)

    @define_scope
    def optimize(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            loss, accuracy, num_correct = self.metrics
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)

            return train_op, loss, accuracy
    
    @define_scope(scope='stream_metrics')
    def metrics(self):
        d = self.get_single_device()
        with tf.device(assign_to_device(d, self.config.controller)):
            query = self.prediction
            loss, accuracy, num_correct = prototypical_networks_loss(self.p, query, tf.shape(self.query)[1], self.label)
            return loss, accuracy, num_correct
    
