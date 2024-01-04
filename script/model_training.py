# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 21:38:20 2023

@author: ahmad104
"""

import pickle
import multiprocessing
import os

from training_helpers import get_train_test_val, loss, model_checkpoint
from model_creation import create_siamese_network
from data_visualizers import display_model_info, visualize
from image_loader import DataGenerator

from utils import MODELS_TYPES, MODEL_SAVE_PATH, HISTORY_SAVE_PATH, MONITORING_METRIC, BATCH_SIZE, EPOCHS, MARGIN
    
def run_training_process(model: MODELS_TYPES):
    pairs_train, labels_train, pairs_val, labels_val, pairs_test, labels_test = get_train_test_val()

    siamese = create_siamese_network(model)
    siamese.compile(loss=loss(margin=MARGIN), optimizer="RMSprop", metrics=[MONITORING_METRIC])
    siamese.summary()

    model_path = MODEL_SAVE_PATH.format(model_name = siamese.name)
    
    # if os.path.exists(f'{model_path}.index'):
    #     display_model_info(siamese, model_path, pairs_test, labels_test)
    #     return
    
    visualize(pairs_train[:-1], labels_train[:-1])
    visualize(pairs_val[:-1], labels_val[:-1])
    visualize(pairs_test[:-1], labels_test[:-1])
    
    
    history = siamese.fit(
        DataGenerator(pairs_train, labels_train, BATCH_SIZE),
        validation_data=DataGenerator(pairs_val, labels_val, BATCH_SIZE),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=model_checkpoint(model_path, MONITORING_METRIC),
    )
    
    siamese.save_weights(model_path)
    
    with open(HISTORY_SAVE_PATH.format(model_name = siamese.name), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == "__main__":
    run_training_process(MODELS_TYPES.ONE_SHOT_LEARNING.value)
    # for model in MODELS_TYPES:
    #     if model == MODELS_TYPES.KERAS_SIAMESE_CONTRASTIVE:
    #         continue
    #     run_training_process(model.value)
        # p = multiprocessing.Process(target=run_training_process, kwargs={"model": model.value})
        # p.start()
        # p.join()