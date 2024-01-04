# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:30:40 2023

@author: ahmad104
"""

from image_loader import fetch_img

import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from keras.callbacks import History

from utils import BATCH_SIZE, HISTORY_SAVE_PATH, TO_SHOW, NUM_COL
from image_loader import DataGenerator

def visualize(pairs, labels, predictions=None, test=False):
    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = TO_SHOW // NUM_COL if TO_SHOW // NUM_COL != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * NUM_COL

    # Plot the images
    fig, axes = plt.subplots(num_row, NUM_COL, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % NUM_COL]
        else:
            ax = axes[i // NUM_COL, i % NUM_COL]

        ax.imshow(tf.concat([fetch_img(pairs[i][0]), fetch_img(pairs[i][1])], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

def plt_metric(history, metric, title, has_valid=True):
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()

def display_history_plots(history):
    # Plot the accuracy
    plt_metric(history=history.history, metric="accuracy", title="Model accuracy")
    
    # Plot the contrastive loss
    plt_metric(history=history.history, metric="loss", title="Contrastive Loss")

def display_test_data(model, pairs_test, labels_test):
    results = model.evaluate(DataGenerator(pairs_test, labels_test, BATCH_SIZE))
    print("test loss, test acc:", results)
    
    predictions = model.predict(DataGenerator(pairs_test, labels_test, BATCH_SIZE))
    visualize(pairs_test, labels_test, predictions=predictions, test=True)

def display_model_info(model, model_path, pairs_test, labels_test):
    history = History()
    
    with open(HISTORY_SAVE_PATH.format(model_name = model.name), "rb") as file_pi:
        history.history = pickle.load(file_pi)
    
    print(f'\n\n------------------FOR MODEL: {model.name}------------------\n\n')
    
    display_history_plots(history)
    display_test_data(model, pairs_test, labels_test)