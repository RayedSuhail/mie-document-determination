# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 21:38:20 2023

@author: ahmad104
"""

import pickle

from image_loader import get_all_imgs_dictlist

from training_helpers import get_train_test_val, loss, DataGenerator, model_checkpoint

from model_creation import MODELS_LIST, create_siamese_network

from data_visualizers import plt_metric, visualize

from keras.callbacks import History

LABEL_PATHS = {
    'training_labels': '../dataset/labels/train.txt',
    'testing_labels': '../dataset/labels/test.txt',
    'validation_labels': '../dataset/labels/val.txt',
}

MODEL_SAVE_PATH = '../models/{model_name}/model_final'
HISTORY_SAVE_PATH = '../results/{model_name}_history'
MONITORING_METRIC = 'val_loss'

EPOCHS = 20
BATCH_SIZE = 16
MARGIN = 1  # Margin for contrastive loss.

all_imgs = get_all_imgs_dictlist(LABEL_PATHS)

pairs_train, labels_train, pairs_val, labels_val, pairs_test, labels_test = get_train_test_val(all_imgs)
    
def run_training_process(only_models = False):
    for model in MODELS_LIST:
        siamese = create_siamese_network(model, 'Siamese_Model')
        siamese.compile(loss=loss(margin=MARGIN), optimizer="RMSprop", metrics=["accuracy"])
        siamese.summary()
        
        if only_models:
            display_model_info(siamese)
            continue
        
        visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)
        visualize(pairs_val[:-1], labels_val[:-1], to_show=4, num_col=4)
        visualize(pairs_test[:-1], labels_test[:-1], to_show=4, num_col=4)
        
        
        history = siamese.fit(
            DataGenerator(pairs_train, labels_train, BATCH_SIZE),
            validation_data=DataGenerator(pairs_val, labels_val, BATCH_SIZE),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=model_checkpoint(MODEL_SAVE_PATH.format(model_name = siamese.name), MONITORING_METRIC),
        )
        
        siamese.save_weights(MODEL_SAVE_PATH.format(model_name = siamese.name))
        
        with open(HISTORY_SAVE_PATH.format(model_name = siamese.name), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

def display_history_plots(history):
    # Plot the accuracy
    plt_metric(history=history.history, metric="accuracy", title="Model accuracy")
    
    # Plot the contrastive loss
    plt_metric(history=history.history, metric="loss", title="Contrastive Loss")

def display_test_data(model):
    results = model.evaluate(DataGenerator(pairs_test, labels_test, BATCH_SIZE))
    print("test loss, test acc:", results)
    
    predictions = model.predict(DataGenerator(pairs_test, labels_test, BATCH_SIZE))
    visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)

def display_model_info(model):
    model.load_weights(MODEL_SAVE_PATH.format(model_name = model.name))
    
    history = History()
    
    with open(HISTORY_SAVE_PATH.format(model_name = model.name), "rb") as file_pi:
        history.history = pickle.load(file_pi)
    
    print(f'\n\n------------------FOR MODEL: {model.name}------------------\n\n')
    
    display_history_plots(history)
    display_test_data(model)

run_training_process(True)