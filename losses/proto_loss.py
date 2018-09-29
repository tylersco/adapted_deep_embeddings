'''
Adapted from Jake Snell's implementation (https://github.com/jakesnell/prototypical-networks)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def prototypical_networks_loss(prototypes, query, num_query_per_class, target_indices):
    m = tf.shape(prototypes)[0]

    prototypes = tf.expand_dims(prototypes, 0)
    query = tf.expand_dims(query, 1)

    dist = tf.reduce_sum(tf.pow(query - prototypes, 2), 2)
    log_prob = tf.nn.log_softmax(-dist)
    log_prob = tf.reshape(log_prob, shape=[m, num_query_per_class, -1])

    idx1 = tf.reshape(tf.range(0, m), shape=[m, 1, 1])
    idx1 = tf.reshape(tf.tile(idx1, multiples=(1, num_query_per_class, 1)), shape=[-1])
    idx1 = tf.expand_dims(idx1, 1)

    idx2 = tf.expand_dims(tf.tile(tf.range(0, num_query_per_class), multiples=(m,)), 1)

    indices = tf.concat([idx1, idx2, idx1], axis=1)

    loss = tf.squeeze(tf.gather_nd(-log_prob, indices))
    loss = tf.reduce_mean(tf.reshape(loss, shape=[-1]))

    y_hat = tf.argmax(log_prob, 2)
    correct = tf.equal(y_hat, target_indices)
    num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return loss, accuracy, num_correct
