'''
Credit: Karl Ridgeway (CU Boulder Computer Science PhD)
Adapted: Tyler Scott (CU Boulder Computer Science Master's)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections

def generate_for_batch(batch_ids):
    pos_comps=[]
    neg_comps=[]
    unique_batch_ids, unique_batch_id_counts = np.unique(batch_ids, return_counts=True)
    for batch_idx in range(len(batch_ids)):
        batch_id = batch_ids[batch_idx]
        negatives = np.where(batch_ids != batch_id)[0]
        positives = np.where(batch_ids == batch_id)[0]
        these_pos_comps = map(lambda idx: (batch_idx, idx), filter(lambda idx: idx > batch_idx, positives) )
        these_neg_comps = map(lambda idx: (batch_idx, idx), filter(lambda idx: idx > batch_idx, negatives) )
        pos_comps.extend(these_pos_comps)
        neg_comps.extend(these_neg_comps)
    pos_comps = np.array(pos_comps)
    neg_comps = np.array(neg_comps)
    return pos_comps, neg_comps

# Generates the "full" combinations of pairs for each minibatch
def generate_batch_pairs(ids, batch_size, examples_per_identity):
    batch_number=0
    global id_indices, unique_ids
    unique_ids = np.unique(ids)
    id_indices=collections.defaultdict(list)
    for idx,identity in enumerate(ids):
        id_indices[identity].append(idx)
    def randomize_order():
        global id_indices, unique_ids
        unique_ids = np.random.permutation(unique_ids)
        n_identities_left=0
        ids_to_remove = []
        for unique_id in unique_ids:
            if len(id_indices[unique_id]) > 1:
                n_identities_left+=1
            else:
                ids_to_remove.append(unique_id)
            id_indices[unique_id] = np.random.permutation(id_indices[unique_id])
        unique_ids = unique_ids[ np.logical_not( np.in1d(unique_ids, ids_to_remove) ) ]
        return n_identities_left
    randomize_order()
    training=True
    cur_identity_idx=0
    n_samp_per_id = examples_per_identity
    while training:
        if cur_identity_idx >= len(unique_ids):
            cur_identity_idx = 0
            if randomize_order() < 2:
                training = False
                continue
        batch_ids=[]
        batch_samples=[]
        for identity in unique_ids[cur_identity_idx:cur_identity_idx+batch_size]:
            samples = id_indices[identity][:n_samp_per_id]
            id_indices[identity] = np.delete(id_indices[identity], np.arange(n_samp_per_id))
            batch_ids.extend( [identity] * len(samples) )
            batch_samples.extend(samples)
        cur_identity_idx = cur_identity_idx + batch_size
        pos_comps, neg_comps = generate_for_batch(batch_ids)
        batch_number+=1
        if len(pos_comps) > 0 and len(neg_comps) > 0:
            return {'batch_samples':batch_samples,
                'batch_ids': batch_ids,
                'pos_comps':pos_comps,
                'neg_comps':neg_comps,
                'batch_idx':batch_number,
                }
        batch_samples=[]
        batch_ids=[]
