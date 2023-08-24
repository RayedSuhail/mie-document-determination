# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:02:04 2023

@author: ahmad104
"""
import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Add, Input
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Lambda
from keras.regularizers import l2

INPUT_SHAPE = (100, 75, 1)
NUM_CLASSES = 16
MODELS_LIST = ['Siamese_Contrastive', 'OneShot', 'AlexNet', 'VGGNet', 'ResNet']

def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def create_siamese_network(base_model: str, model_name: str) -> keras.engine.functional.Functional:
    if base_model not in MODELS_LIST:
        raise TypeError('Wrong Model ID string')

    if base_model == 'Siamese_Contrastive':
        model = SC_CNN()
    if base_model == 'OneShot':
        model = OS_CNN()
    if base_model == 'AlexNet':
        model = AlexNet_CNN()
    if base_model == 'VGGNet':
        model = VGGNet_CNN()
    if base_model == 'ResNet':
        model = ResNet_CNN()
    
    input_l = Input(INPUT_SHAPE)
    input_r = Input(INPUT_SHAPE)

    model_l = model(input_l)
    model_r = model(input_r)

    interim_model = Lambda(euclidean_distance)([model_l, model_r])
    interim_model = BatchNormalization()(interim_model)
    interim_model = Dense(1, activation = 'sigmoid')(interim_model)

    siamese_model = Model(inputs = [input_l, input_r], outputs = interim_model, name = f'{base_model}_{model_name}')
    return siamese_model

def SC_CNN() -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = BatchNormalization()(input_x)
    X = Conv2D(4, (5, 5), activation="tanh")(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    X = Conv2D(16, (5, 5), activation="tanh")(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    X = Flatten()(X)
    
    X = BatchNormalization()(X)
    X = Dense(10, activation="tanh")(X)
    model = Model(inputs = input_x, outputs = X, name = 'Siamese_Contrastive')

    return model

def OS_CNN() -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = BatchNormalization()(input_x)
    X = Conv2D(64, (10,10), activation='relu', kernel_regularizer=l2(2e-4))(X)
    X = MaxPooling2D()(X)

    X = Conv2D(128, (7,7), activation='relu', kernel_regularizer=l2(2e-4))(X)
    X = MaxPooling2D()(X)

    X = Conv2D(128, (4,4), activation='relu', kernel_regularizer=l2(2e-4))(X)
    X = MaxPooling2D()(X)

    X = Conv2D(256, (4,4), activation='relu', kernel_regularizer=l2(2e-4))(X)

    X = Flatten()(X)
    X = Dense(4096, activation='relu', kernel_regularizer=l2(1e-3))(X)

    X = Dense(NUM_CLASSES, activation= 'softmax')(X)

    model = Model(inputs = input_x, outputs = X, name = 'OneShot')

    return model

def AlexNet_CNN() -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = BatchNormalization()(input_x)
    X = Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        kernel_initializer = 'he_normal')(X)
    X = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(X)

    X = Conv2D(256, kernel_size=(5,5), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer = 'he_normal')(X)
    X = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(X)

    X = Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer = 'he_normal')(X)

    X = Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer = 'he_normal')(X)

    X = Conv2D(256, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer = 'he_normal')(X)

    X = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format= None)(X)
    
    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(4096, activation= 'relu')(X)
    X = Dropout(0.5)(X)

    X = Dense(4096, activation= 'relu')(X)
    X = Dense(1000, activation= 'relu')(X)

    X = Dense(NUM_CLASSES, activation= 'softmax')(X)

    model = Model(inputs = input_x, outputs = X, name = 'AlexNet')
    
    return model

def VGGNet_CNN() -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = BatchNormalization()(input_x)
    X = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(X)
    X = MaxPooling2D((2, 2))(X)

    X = Flatten()(X)
    X = Dense(4096, activation="relu")(X)
    X = Dense(4096, activation="relu")(X)

    X = Dense(NUM_CLASSES, activation="softmax")(X)

    model = Model(inputs = input_x, outputs = X, name = 'VGGNet16')

    return model

def identity_block(X, f, filters, stage, block) -> keras.engine.functional.Functional:
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X
   
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2) -> keras.engine.functional.Functional:
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet_CNN() -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(input_x)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    X = Flatten()(X)
    X = Dense(256, activation='relu', name='fc1')(X)
    X = Dense(128, activation='relu', name='fc2')(X)
    
    X = Dense(NUM_CLASSES, activation='softmax', name='fc3')(X)
    
    model = Model(inputs=input_x, outputs = X, name = 'ResNet50')

    return model