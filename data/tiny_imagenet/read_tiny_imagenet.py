import os
import sys
import random
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize

class TinyImageNet():
    def __init__(self, path):
        self.path = path

        self.images = []
        self.labels = []

    def load_data(self):
        ims, labels = self.load(self.path)

        self.images = self.process_images(ims)
        self.labels = self.process_labels(labels)

        return self.images, self.labels

    def process_images(self, images):
        images_np = np.array(images) / 255.0
        return images_np

    def process_labels(self, labels):
        return np.array(labels)

    @classmethod
    def load(cls, path):
        class_id = 0
        id_to_label = {}
        validation_annotations = None
        validation_images = {}
        images = []
        labels = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f == 'val_annotations.txt':
                    validation_annotations = os.path.join(root, f)
                elif f.endswith('.JPEG'):
                    path = os.path.join(root, f)
                    id = f.split('_')[0]
                    if id == 'val':
                        validation_images[f] = path
                    else:
                        if id not in id_to_label:
                            id_to_label[id] = class_id
                            class_id += 1
                        img = imread(path)
                        if len(img.shape) == 2:
                            img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                        images.append(img)
                        labels.append(id_to_label[id])

        with open(validation_annotations) as val_ann:
            for line in val_ann:
                contents = line.split()
                img = imread(validation_images[contents[0]])
                if len(img.shape) == 2:
                    img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
                images.append(img)
                labels.append(id_to_label[contents[1]])

        return images, labels

    def kntl_data_form(self, k1, n1, k2, n2):
        assert n1 + n2 <= 200
        assert k1 < 550 and k2 < 550
        self.load_data()

        print('Full dataset: {0}'.format(len(self.labels)))

        all_classes = np.unique(self.labels)
        print('Number of classes: {0}'.format(len(all_classes)))

        task2_classes = np.sort(np.random.choice(all_classes, n2, replace=False))
        all_classes = np.delete(all_classes, np.where(np.isin(all_classes, task2_classes)))
        indices = np.isin(self.labels, task2_classes)
        self.x_task2, self.y_task2 = self.images[indices], self.labels[indices]
        shuffle = np.random.permutation(len(self.y_task2))
        self.x_task2, self.y_task2 = self.x_task2[shuffle], self.y_task2[shuffle]

        task1_classes = np.sort(np.random.choice(all_classes, n1, replace=False))
        indices = np.isin(self.labels, task1_classes)
        self.x_task1, self.y_task1 = self.images[indices], self.labels[indices]
        shuffle = np.random.permutation(len(self.y_task1))
        self.x_task1, self.y_task1 = self.x_task1[shuffle], self.y_task1[shuffle]

        print('Task 1 Full: {0}'.format(len(self.y_task1)))
        print('Task 2 Full: {0}\n'.format(len(self.y_task2)))

        # Force class labels to start from 0 and increment upwards by 1
        sorted_class_indices = np.sort(np.unique(self.y_task1))
        zero_based_classes = np.arange(0, len(sorted_class_indices))
        for i in range(len(self.y_task1)):
            self.y_task1[i] = zero_based_classes[sorted_class_indices == self.y_task1[i]]

        self.x_train_task1 = []
        self.y_train_task1 = []
        self.x_valid_task1 = []
        self.y_valid_task1 = []

        for i in zero_based_classes:
            all_indices = np.where(self.y_task1 == i)[0]
            idx = np.random.choice(all_indices, k1, replace=False)
            self.x_train_task1.extend(self.x_task1[idx])
            self.y_train_task1.extend(self.y_task1[idx])
            all_indices = np.delete(all_indices, np.where(np.isin(all_indices, idx)))
            self.x_valid_task1.extend(self.x_task1[all_indices])
            self.y_valid_task1.extend(self.y_task1[all_indices])

        self.x_train_task1 = np.array(self.x_train_task1)
        self.y_train_task1 = np.array(self.y_train_task1)
        self.x_valid_task1 = np.array(self.x_valid_task1)
        self.y_valid_task1 = np.array(self.y_valid_task1)

        print('Task 1 training: {0}'.format(len(self.x_train_task1)))
        print('Task 1 validation: {0}'.format(len(self.x_valid_task1)))

        # Force class labels to start from 0 and increment upwards by 1
        sorted_class_indices = np.sort(np.unique(self.y_task2))
        zero_based_classes = np.arange(0, len(sorted_class_indices))
        for i in range(len(self.y_task2)):
            self.y_task2[i] = zero_based_classes[sorted_class_indices == self.y_task2[i]]

        self.x_train_task2 = []
        self.y_train_task2 = []
        for i in zero_based_classes:
            idx = np.random.choice(np.where(self.y_task2 == i)[0], k2, replace=False)
            self.x_train_task2.extend(self.x_task2[idx])
            self.y_train_task2.extend(self.y_task2[idx])
            self.x_task2 = np.delete(self.x_task2, idx, axis=0)
            self.y_task2 = np.delete(self.y_task2, idx, axis=0)

        self.x_train_task2 = np.array(self.x_train_task2)
        self.y_train_task2 = np.array(self.y_train_task2)

        k_test = 550 - k2

        self.x_test_task2 = []
        self.y_test_task2 = []
        for i in zero_based_classes:
            idx = np.random.choice(np.where(self.y_task2 == i)[0], k_test, replace=False)
            self.x_test_task2.extend(self.x_task2[idx])
            self.y_test_task2.extend(self.y_task2[idx])

        self.x_test_task2 = np.array(self.x_test_task2)
        self.y_test_task2 = np.array(self.y_test_task2)

        print('k = {0}, n = {1}'.format(k2, n2))
        print('Task 2 training: {0}'.format(len(self.x_train_task2)))
        print('Task 2 test: {0}\n'.format(len(self.x_test_task2)))

        return (self.x_train_task1, self.y_train_task1), (self.x_valid_task1, self.y_valid_task1), (self.x_train_task2, self.y_train_task2), (self.x_test_task2, self.y_test_task2)
