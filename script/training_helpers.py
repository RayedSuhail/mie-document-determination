# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:24:19 2023

@author: ahmad104
"""

from image_loader import get_all_imgs_dictlist, sample_random_imgs

import random
import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import LABEL_PATHS, NUM_CLASSES

def model_checkpoint(save_dir: str, metric: str):
    early_stopping = EarlyStopping(monitor=metric,
                              min_delta=0,
                              patience=3,
                              verbose=1,
                              restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_dir, 
                monitor=metric, verbose=1, 
                save_best_only=True)
    return [early_stopping]

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
    
    digit_indices = [np.where(np.array(y) == i)[0] for i in range(NUM_CLASSES)]

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
        label2 = random.randint(0, NUM_CLASSES - 1)
        while label2 == label1:
            label2 = random.randint(0, NUM_CLASSES - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")

def get_train_test_val():
    all_imgs = get_all_imgs_dictlist()

    train_data = sample_random_imgs(LABEL_PATHS.TRAINING.name, all_imgs)
    val_data = sample_random_imgs(LABEL_PATHS.VALIDATION.name, all_imgs)
    test_data = sample_random_imgs(LABEL_PATHS.TESTING.name, all_imgs)

    pairs_train, labels_train = make_pairs(train_data)
    pairs_val, labels_val = make_pairs(val_data)
    pairs_test, labels_test = make_pairs(test_data)
    
    return(pairs_train, labels_train, pairs_val, labels_val, pairs_test, labels_test)
