# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:24:19 2023

@author: ahmad104
"""

from image_loader import fetch_img

import math
import random
import numpy as np
import tensorflow as tf

TOTAL_TRAINING = 32000
TOTAL_TESTING = 4000
TOTAL_VALIDATION = 4000

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        return (np.array([fetch_img(pair[0]) for pair in batch_x]), np.array([fetch_img(pair[1]) for pair in batch_x])), np.array(batch_y)

def loss(margin=1):
    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

def make_pairs(all_imgs):
    x, y = list(), list()
    for img in all_imgs:
        x.append(img['path'])
        y.append(img['category'])
    
    num_classes = max(y) + 1
    digit_indices = [np.where(np.array(y) == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")

def get_train_test_val(all_imgs):
    train_data = random.sample(list(filter(lambda img: img['type'] == 'training_labels', all_imgs)), TOTAL_TRAINING)
    val_data = random.sample(list(filter(lambda img: img['type'] == 'validation_labels', all_imgs)), TOTAL_TESTING)
    test_data = random.sample(list(filter(lambda img: img['type'] == 'testing_labels', all_imgs)), TOTAL_VALIDATION)

    pairs_train, labels_train = make_pairs(train_data)

    pairs_val, labels_val = make_pairs(val_data)

    pairs_test, labels_test = make_pairs(test_data)
    
    return(pairs_train, labels_train, pairs_val, labels_val, pairs_test, labels_test)
