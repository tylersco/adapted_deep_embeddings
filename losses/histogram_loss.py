'''
Credit: Karl Ridgeway (CU Boulder)
Adapted: Tyler Scott (CU Boulder)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import data.batch_pair_generator as gen

# Estimate the histogram using the assigments of points to grid bins
def getDistributionDensity(x, n, grid, grid_delta):
    grid_size = grid.get_shape().as_list()[0]
    def process_grid_cell(i):
        def get_left_add():
            inds = tf.reshape( tf.where(tf.logical_and(x >= grid[i-1], x < grid[i]) ), (-1,) )
            left_dist = tf.gather(x, inds)
            left_add = tf.reduce_sum(left_dist - grid[i-1])
            return left_add
        def get_right_add():
            inds = tf.reshape(tf.where(tf.logical_and(x >= grid[i], x < grid[i+1])), (-1,) )
            right_dist = tf.gather(x, inds)
            right_add = tf.reduce_sum(grid[i+1] - right_dist)
            return right_add
        left_add = tf.cond( i > 0, get_left_add, lambda: tf.constant(0.0, dtype=tf.float32))
        right_add = tf.cond( i < grid_size-1, get_right_add, lambda: tf.constant(0.0, dtype=tf.float32))
        return left_add + right_add
    p_list = tf.map_fn(process_grid_cell, np.arange(grid_size, dtype=np.int32), dtype=tf.float32)
    p = tf.concat(p_list, axis=0)
    p = p / ( tf.cast(n, tf.float32) * grid_delta)
    return p

# Calculates probability of wrong order in pairs' similarities: positive pair less similar than negative one
# (this corresponds to 'simple' loss, other variants ('linear', 'exp') are generalizations that take into account
# not only the order but also the difference between the two similarity values).
# Can use histogram and beta-distribution to fit input data.
def hist_loss(z, pos_comps, n_pos_comps, neg_comps, n_neg_comps):
    grid_delta = 0.01
    grid_arr = np.array([i for i in np.arange(-1., 1. + grid_delta, grid_delta)])
    grid_len = len(grid_arr)
    grid = tf.constant(grid_arr, dtype=tf.float32)
    L = np.ones((grid_len, grid_len))
    for i in range(grid_len):
        L[i] = grid_arr[i] <= grid_arr
    L = tf.constant(L, dtype=tf.float32)

    pos_comps_l = tf.reshape(pos_comps[:,0], (-1,))
    d_pos = tf.reduce_sum( tf.multiply(tf.gather(z,pos_comps[:,0]), tf.gather(z,pos_comps[:,1])), axis=1)
    d_neg = tf.reduce_sum( tf.multiply(tf.gather(z,neg_comps[:,0]), tf.gather(z,neg_comps[:,1])), axis=1)

    distr_pos = getDistributionDensity(d_pos, n_pos_comps, grid, grid_delta)
    distr_neg = getDistributionDensity(d_neg, n_neg_comps, grid, grid_delta)
    distr_pos = tf.reshape(distr_pos, (1,-1))
    distr_neg = tf.reshape(distr_neg, (-1,1))
    result = tf.matmul( tf.matmul(distr_pos, L), distr_neg )
    return result

def train_batches(X_train, ids, batch_size):
    batch_idx=0
    examples_per_id = int(batch_size / len(np.unique(ids)))
    if examples_per_id == 1:
        examples_per_id += 1
    for batch in gen.generate_batch_pairs(ids, batch_size, examples_per_id):
        feed_dict = {'x':X_train[batch['batch_samples']],
                         'pos_comps':batch['pos_comps'],
                         'neg_comps':batch['neg_comps'],
                         'n_pos_comps':len(batch['pos_comps']),
                         'n_neg_comps':len(batch['neg_comps'])
        }
        yield batch['batch_idx'], feed_dict
