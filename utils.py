import numpy as np
import tensorflow as tf

from data.episode_generator import generate_training_episode, generate_evaluation_episode

def classification_batch_evaluation(sess, model, ops, batch_size, is_task1, x, y=None, stream=False):
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='stream_metrics')
    sess.run(tf.variables_initializer(var_list=running_vars))

    result = []
    for i in range(0, len(y), batch_size):
        x_mb, y_mb = x[i:i + batch_size], y[i:i + batch_size]
        result.append(sess.run(ops, feed_dict={model.input: x_mb, model.target: y_mb, model.is_task1: is_task1, model.is_train: False}))
    
    if stream:
        return result[-1]
    return result

def hist_loss_batch_eval(sess, model, ops, batch_size, x):
    hidden_rep = None
    for i in range(0, len(x), batch_size):
        x_mb = x[i:i + batch_size]
        h = sess.run(ops, feed_dict={model.input: x_mb, model.is_train: False})
        if hidden_rep is None:
            hidden_rep = h
        else:
            hidden_rep = np.concatenate((hidden_rep, h), axis=0)
    return hidden_rep

def proto_episodic_performance(sess, model, x, y, num_classes, num_support, num_query, batch_size, evaluation_episodes):
    perf = []
    total_cost = 0.0
    total_accuracy = 0.0
    total_query = 0

    for support_batch, query_batch, query_labels_batch in generate_training_episode(x, y, num_classes, num_support, num_query, evaluation_episodes, batch_size=batch_size):
        feed_dict = {
            model.query: query_batch,
            model.label: query_labels_batch,
            model.is_train: False
        }
        
        if model.config.dataset == 'tiny_imagenet':
            prototypes = model.compute_batch_prototypes(sess, support_batch, model.config.classes_per_episode)
            feed_dict[model.p] = prototypes
        else:
            feed_dict[model.support] = support_batch
        
        c, acc, num_corr = sess.run(model.metrics, feed_dict=feed_dict)

        if batch_size > 0:
            total_query += query_labels_batch.shape[0] * query_labels_batch.shape[1]
            total_cost += c
            total_accuracy += float(num_corr)
        else:
            perf.append([acc, c])

    if batch_size > 0:
        total_accuracy /= total_query
        total_cost /= (evaluation_episodes * batch_size)
        perf.append([total_accuracy, total_cost])

    return np.mean(perf, axis=0), np.std(perf, axis=0)

def proto_performance(sess, model, x_s, y_s, x_q, y_q, batch_size):
    total_cost = 0.0
    total_accuracy = 0.0
    num_query = 0

    # Generator will only be run once if batch_size <= 0
    for support_batch, query_batch, query_labels_batch in generate_evaluation_episode(x_s, y_s, x_q, y_q, batch_size=batch_size):
        feed_dict = {
                model.query: query_batch,
                model.label: query_labels_batch,
                model.is_train: False
        }
        
        if model.config.dataset == 'tiny_imagenet':
            prototypes = model.compute_batch_prototypes(sess, support_batch, model.config.classes_per_episode)
            feed_dict[model.p] = prototypes
        else:
            feed_dict[model.support] = support_batch
        
        c, acc, num_corr = sess.run(model.metrics, feed_dict=feed_dict)

        if batch_size > 0:
            num_query += query_labels_batch.shape[0] * query_labels_batch.shape[1]
            total_cost += c
            total_accuracy += float(num_corr)
        else:
            total_cost = c
            total_accuracy = acc 
    
    if batch_size > 0:
        total_accuracy /= num_query

    return total_cost, total_accuracy
