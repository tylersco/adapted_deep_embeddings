import numpy as np
import sys

def generate_training_episode(x, y, num_classes_per_episode, num_support_per_class, num_query_per_class, num_episodes, batch_size=0):
    all_unique_classes = np.unique(y)
    assert num_classes_per_episode <= len(all_unique_classes)

    data_by_class = {}

    for c in all_unique_classes:
        d = x[y == c]
        d_y = y[y == c]
        data_by_class[c] = (d, len(d), d_y)

    unique_classes = np.unique(y)

    for i in range(num_episodes):
        support_batch = []
        query_batch = []
        query_labels_batch = []

        try:
            episode_classes = np.sort(np.random.choice(unique_classes, num_classes_per_episode, replace=False))
        except ValueError as e:
            print(e)
            return

        for class_label in episode_classes:
            try:
                idx = np.random.choice(data_by_class[class_label][1], num_support_per_class, replace=False)
                idx_set = set(idx)
                idx_complement = []
                for i in range(data_by_class[class_label][1]):
                    if i not in idx_set:
                        idx_complement.append(i)

                support_data = data_by_class[class_label][0][idx]

                idx = np.random.choice(idx_complement, num_query_per_class, replace=False)
                query_data = data_by_class[class_label][0][idx]
                query_labels = data_by_class[class_label][2][idx]

                support_batch.append(support_data)
                query_batch.append(query_data)
                query_labels_batch.append(query_labels)
            except ValueError as e:
                print(e)
                return

        support_batch = np.array(support_batch)
        query_batch = np.array(query_batch)
        query_labels_batch = np.array(query_labels_batch)

        if batch_size > 0:
            query_batches = np.array_split(query_batch, int(np.ceil(len(query_batch[0]) / float(batch_size))), axis=1)
            query_label_batches = np.array_split(query_labels_batch, int(np.ceil(len(query_labels_batch[0]) / float(batch_size))), axis=1)
            for i in range(len(query_batches)):
                yield support_batch, query_batches[i], query_label_batches[i]
        else:
            yield support_batch, query_batch, query_labels_batch

def generate_evaluation_episode(x_support, y_support, x_query, y_query, batch_size=0):
    assert len(np.unique(y_support)) == len(np.unique(y_query))
    all_unique_classes = np.unique(y_query)
    min_support = float('inf')
    min_query = float('inf')

    data_by_class = {}

    for c in all_unique_classes:
        d_support = x_support[y_support == c]
        if min_support > len(d_support):
            min_support = len(d_support)
        d_query = x_query[y_query == c]
        if min_query > len(d_query):
            min_query = len(d_query)
        d_y_query = y_query[y_query == c]
        data_by_class[c] = (d_support, len(d_support), d_query, len(d_query), d_y_query)

    unique_classes = np.unique(y_query)

    support_batch = []
    query_batch = []
    query_labels_batch = []

    for class_label in unique_classes:
        try:
            idx = np.random.choice(data_by_class[class_label][1], min_support, replace=False)
            support_data = data_by_class[class_label][0][idx]

            idx = np.random.choice(data_by_class[class_label][3], min_query, replace=False)
            query_data = data_by_class[class_label][2][idx]
            query_labels = data_by_class[class_label][4][idx]

            support_batch.append(support_data)
            query_batch.append(query_data)
            query_labels_batch.append(query_labels)
        except ValueError as e:
            print(e)
            return

    support_batch = np.array(support_batch)
    query_batch = np.array(query_batch)
    query_labels_batch = np.array(query_labels_batch)

    if batch_size > 0:
        query_batches = np.array_split(query_batch, int(np.ceil(len(query_batch[0]) / float(batch_size))), axis=1)
        query_label_batches = np.array_split(query_labels_batch, int(np.ceil(len(query_labels_batch[0]) / float(batch_size))), axis=1)
        for i in range(len(query_batches)):
            yield support_batch, query_batches[i], query_label_batches[i]
    else:
        yield support_batch, query_batch, query_labels_batch
