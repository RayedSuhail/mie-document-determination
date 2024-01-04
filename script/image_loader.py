# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:33:44 2023

@author: ahmad104
"""

import random
from PIL import Image
import numpy as np
from typing import List
import tensorflow as tf
import math

from utils import LABEL_PATHS, TOTAL_TRAINING, TOTAL_TESTING, TOTAL_VALIDATION, INPUT_SHAPE, FORBIDDEN_LIST_PATH

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



def read_file_for_record_type(label_path: str) -> List[str]:
    return open(label_path).readlines()

def get_path_and_category(path_category: str) -> (str, str):
    return path_category.split(maxsplit = 1)

def fetch_img(img_path: str) -> np.ndarray:
    try:
        with Image.open(f'../dataset/images/{img_path}') as open_file:
            open_file.load()
            resized = open_file.resize(INPUT_SHAPE[1::-1], Image.ANTIALIAS)
            img = np.asarray(resized)
            open_file.close()
        return img
    except Exception as e:
        print('Error while fetching from: ', img_path, '\n', e)

def sample_random_imgs(img_type: LABEL_PATHS, all_imgs: dict) -> List[np.ndarray]:
    if img_type == LABEL_PATHS.TRAINING.name:
        total_count = TOTAL_TRAINING
    elif img_type == LABEL_PATHS.TESTING.name:
        total_count = TOTAL_TESTING
    elif img_type == LABEL_PATHS.VALIDATION.name:
        total_count = TOTAL_VALIDATION
    return random.sample(list(filter(lambda img: img['type'] == img_type, all_imgs)), total_count)

def get_all_imgs_dictlist() -> List[dict]:
    all_imgs = list()
    forbidden_list = [img.strip() for img in read_file_for_record_type(FORBIDDEN_LIST_PATH)]
    for label_type in LABEL_PATHS:
        for path_category in read_file_for_record_type(label_type.value):
            path, category = get_path_and_category(path_category)
            if path in forbidden_list:
                continue
            all_imgs.append({'path': path, 'category': int(category.strip()), 'type': label_type.name})
    return all_imgs