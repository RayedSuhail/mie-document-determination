# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:33:44 2023

@author: ahmad104
"""

import random
import copy
from PIL import Image
import numpy as np
from typing import List


def generate_labels_for_category(label_path: str) -> List[str]:
    return open(label_path).readlines()

def get_path_and_category(path_category: str) -> (str, str):
    return path_category.split(maxsplit = 1)

def fetch_img(img_path: str) -> np.ndarray:
    try:
        open_file = Image.open(f'../dataset/images/{img_path}')
        file = copy.deepcopy(open_file.resize((75, 100), Image.ANTIALIAS))
        img = np.asarray(file)
        open_file.close()
        return img
    except Exception as e:
        print(e)

def two_random_imgs(img_type: str, all_imgs: dict) -> List[np.ndarray]:
    samples = random.sample(list(filter(lambda img: img['category'] == img_type, all_imgs)), 2)
    return list(fetch_img(img['path']) for img in samples)

def get_all_imgs_dictlist(paths: dict) -> List[dict]:
    all_imgs = list()
    for key in paths:
        for path_category in generate_labels_for_category(paths[key]):
            path, category = get_path_and_category(path_category)
            all_imgs.append({'path': path, 'category': int(category.strip()), 'type': key})
    return all_imgs