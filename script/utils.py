# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:47:53 2023

@author: ahmad104
"""

from enum import Enum, EnumMeta

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

class BaseEnum(Enum, metaclass=MetaEnum):
    pass

# =============================================================================
# Utils for data_visualizers.py
# =============================================================================
TO_SHOW = 6
NUM_COL = 3

# =============================================================================
# Utils for model_creation.py
# =============================================================================
INPUT_SHAPE = (100, 75, 1)
NUM_CLASSES = 16
class MODELS_TYPES(BaseEnum):
    KERAS_SIAMESE_CONTRASTIVE = 'Siamese_Contrastive'
    ONE_SHOT_LEARNING = 'OneShot'
    ALEX_NET = 'AlexNet'
    VGG_NET_16 = 'VGGNet'
    RES_NET_50 = 'ResNet'

# =============================================================================
# Utils for model_training.py
# =============================================================================
MODEL_SAVE_PATH = '../models/{model_name}/model_final'
HISTORY_SAVE_PATH = '../results/{model_name}_history'
MONITORING_METRIC = 'accuracy'

EPOCHS = 1
BATCH_SIZE = 16
MARGIN = 1  # Margin for contrastive loss.

class LABEL_PATHS(BaseEnum):
    TRAINING = '../dataset/labels/train.txt'
    TESTING = '../dataset/labels/test.txt'
    VALIDATION = '../dataset/labels/val.txt'

# =============================================================================
# Utils for training_helpers.py
# =============================================================================
TOTAL_TRAINING = 32000
TOTAL_TESTING = 4000
TOTAL_VALIDATION = 4000