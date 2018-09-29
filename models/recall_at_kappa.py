'''
Recall@Kappa Metric
Author: Karl Ridgeway
Contributor: Tyler Scott
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import manhattan_distances

def find_l2_distances(X,Y):
    intersection = -2.* np.dot(X,Y.T)
    X_sum = np.sum(X**2,axis=1)
    Y_sum = np.sum(Y**2,axis=1)
    XY_sum = X_sum[:, np.newaxis] + Y_sum
    return XY_sum + intersection

def find_cos_distances(X,Y):
    return (1.-np.dot(X, Y.T))/2.0

def max_distances(X,Y, dist_fun):
    results = np.zeros( (X.shape[0], Y.shape[0]), dtype=np.float32 )
    if dist_fun == 'max_l1':
        return cdist(X, Y, 'chebyshev')
    else: raise 'not implemented'

def recall_at_kappa_leave_one_out(test_emb, test_id, kappa, dist):
    unique_ids, unique_counts = np.unique(test_id,return_counts=True)
    unique_ids = unique_ids[unique_counts >= 2]
    good_test_indices = np.in1d(test_id,unique_ids)
    valid_test_embs = test_emb[good_test_indices]
    valid_test_ids = test_id[good_test_indices]
    n_correct_at_k = np.zeros(kappa)
    if dist == 'cos':
        distances = find_cos_distances(valid_test_embs,test_emb)
    elif dist == 'l2':
        distances = find_l2_distances(valid_test_embs, test_emb)
    elif dist == 'l1':
        distances = manhattan_distances(valid_test_embs, test_emb)
    elif dist == 'max_l1' or dist == 'max_l2':
        distances = max_distances(valid_test_embs, test_emb, dist)
    for idx, valid_test_id in enumerate(valid_test_ids):
        k_sorted_indices = np.argsort(distances[idx])[1:]
        first_correct_position = np.where(test_id[k_sorted_indices] == valid_test_id)[0][0]
        if first_correct_position < kappa:
            n_correct_at_k[first_correct_position:] += 1
    return 1.*n_correct_at_k / len(valid_test_ids)

def recall_at_kappa_support_query(x_support, y_support, x_query, y_query, kappa, dist):
    n_correct_at_k = np.zeros(kappa)
    if dist == 'cos':
        distances = find_cos_distances(x_query, x_support)
    elif dist == 'l2':
        distances = find_l2_distances(x_query, x_support)
    elif dist == 'l1':
        distances = manhattan_distances(x_query, x_support)
    elif dist == 'max_l1' or dist == 'max_l2':
        distances = max_distances(x_query, x_support, dist)
    for idx, valid_test_id in enumerate(y_query):
        k_sorted_indices = np.argsort(distances[idx])
        first_correct_position = np.where(y_support[k_sorted_indices] == valid_test_id)[0][0]
        if first_correct_position < kappa:
            n_correct_at_k[first_correct_position:] += 1
    return 1.*n_correct_at_k / len(y_query)
