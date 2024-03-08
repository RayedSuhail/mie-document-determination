# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:02:04 2023

@author: ahmad104
"""
import tensorflow as tf
import numpy as np
import os

import keras
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Add, Input
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Lambda
from keras.regularizers import l2

import keras.applications as ka

from utils import INPUT_SHAPE, NUM_CLASSES, MODELS_TYPES, WEIGHTS_PATH

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

def reduce_size(matrix, target, axis):
    result = np.zeros_like(target)
    mat_axis = matrix.shape[axis]
    target_axis = target.shape[axis]
    increments = mat_axis // target_axis
    reshape_tuple = (*matrix.shape[:axis], -1, increments, *matrix.shape[axis + 1:])
    result = matrix.reshape(reshape_tuple).mean(axis=axis+1, dtype=np.float64)
    return result

def upscale_matrix(matrix, target, axis):
    if(len(matrix.shape) != 4):
        raise ValueError("The matrix should have 4 indices")
    result = np.zeros_like(target)
    mat_axis = matrix.shape[axis]
    target_axis = target.shape[axis]
    multiples = target_axis // mat_axis
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(0, matrix.shape[2], multiples):
                for l in range(matrix.shape[3]):
                    result[i][j][k][l] = matrix[i][j][k][l] / multiples
                    result[i][j][k+1][l] = matrix[i][j][k][l] / multiples
        
    return result

def load_pretrained_weights(model: keras.engine.functional.Functional) -> keras.engine.functional.Functional:
    if model.name == MODELS_TYPES.KERAS_SIAMESE_CONTRASTIVE.value:
        pass
    elif model.name == MODELS_TYPES.ONE_SHOT_LEARNING.value:
        pass
    elif model.name == MODELS_TYPES.ALEX_NET.value:
        weights_dict = np.load(WEIGHTS_PATH.format(model_name=model.name), allow_pickle=True, encoding='bytes').item()
        for op_name in weights_dict:
            weights = weights_dict[op_name]
            if op_name == 'fc6':
                weights[0] = reduce_size(weights[0], model.get_layer(op_name).get_weights()[0], 0)
            if op_name == 'conv1':
                weights[0] = reduce_size(weights[0], model.get_layer(op_name).get_weights()[0], 2)
            if op_name == 'conv2' or op_name == 'conv5' or op_name == 'conv4':
                weights[0] = upscale_matrix(weights[0], model.get_layer(op_name).get_weights()[0], 2)
            model.get_layer(op_name).set_weights(weights)
            model.get_layer(op_name).trainable = False
    
    elif model.name == MODELS_TYPES.VGG_NET_16.value:
        weights_dict = np.load(WEIGHTS_PATH.format(model_name=model.name), allow_pickle=True, encoding='bytes').item()
        for op_name in weights_dict:
            weights = weights_dict[op_name]
            if op_name == 'block1_conv1':
                weights[0] = reduce_size(weights[0], model.get_layer(op_name).get_weights()[0], 2)
            if op_name == 'fc1':
                # 25088 % 3072 = 512
                weights[0] = reduce_size(weights[0], model.get_layer(op_name).get_weights()[0], 0)[:3072]
            model.get_layer(op_name).set_weights(weights)
            model.get_layer(op_name).trainable = False
    
    elif model.name == MODELS_TYPES.RES_NET_50.value:
        weights_dict = np.load(WEIGHTS_PATH.format(model_name=model.name), allow_pickle=True, encoding='bytes').item()
        for op_name in weights_dict:
            weights = weights_dict[op_name]
            if op_name == 'conv1_conv':
                weights[0] = reduce_size(weights[0], model.get_layer(op_name).get_weights()[0], 2)
            model.get_layer(op_name).set_weights(weights)
            model.get_layer(op_name).trainable = False
    else:
        raise TypeError('Wrong Model ID string')
    
    return model

def create_model_weights(model_name: str):
    if model_name == MODELS_TYPES.KERAS_SIAMESE_CONTRASTIVE.value:
        return
    elif model_name == MODELS_TYPES.ONE_SHOT_LEARNING.value:
        return
    elif model_name == MODELS_TYPES.ALEX_NET.value:
        tf.keras.utils.get_file(fname=os.path.abspath(WEIGHTS_PATH.format(model_name=model_name)),
                                origin='https://www.cs.toronto.edu/%7Eguerzhoy/tf_alexnet/bvlc_alexnet.npy')
        return
    elif model_name == MODELS_TYPES.VGG_NET_16.value:
        weight_model = ka.vgg16.VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
    elif model_name == MODELS_TYPES.RES_NET_50.value:
        weight_model = ka.resnet.ResNet50(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
    if not (model_name == MODELS_TYPES.ALEX_NET.value and model_name == MODELS_TYPES.KERAS_SIAMESE_CONTRASTIVE.value):
        weight_dict = {}
        for layer in weight_model.layers:
            if (len(weight_model.get_layer(layer.name).get_weights()) != 0) and ('predictions' not in layer.name):
                weight_dict[layer.name] = weight_model.get_layer(layer.name).get_weights()
        np.save(WEIGHTS_PATH.format(model_name=model_name), weight_dict)

def create_siamese_network(base_model: str) -> keras.engine.functional.Functional:
    if base_model == MODELS_TYPES.KERAS_SIAMESE_CONTRASTIVE.value:
        model = SC_CNN()
    elif base_model == MODELS_TYPES.ONE_SHOT_LEARNING.value:
        model = OS_CNN()
    elif base_model == MODELS_TYPES.ALEX_NET.value:
        model = AlexNet_CNN()
    elif base_model == MODELS_TYPES.VGG_NET_16.value:
        model = VGGNet_CNN()
    elif base_model == MODELS_TYPES.RES_NET_50.value:
        model = ResNet_CNN()
    else:
        raise TypeError('Wrong Model ID string')
        
    if not os.path.exists(WEIGHTS_PATH.format(model_name=model.name)):
        try:  
            os.mkdir(os.path.abspath('../models/trained_models'))  
        except OSError as error:  
            print(error) 
        create_model_weights(model.name)
    
    model = load_pretrained_weights(model)
    
    input_l = Input(INPUT_SHAPE)
    input_r = Input(INPUT_SHAPE)

    model_l = model(input_l)
    model_r = model(input_r)

    interim_model = Lambda(euclidean_distance)([model_l, model_r])
    interim_model = BatchNormalization()(interim_model)
    interim_model = Dense(1, activation='sigmoid')(interim_model)

    siamese_model = Model(inputs=[input_l, input_r], outputs=interim_model, name=f'delete/{base_model}_Siamese')
    return siamese_model

def SC_CNN(model_name: str = MODELS_TYPES.KERAS_SIAMESE_CONTRASTIVE.value) -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = BatchNormalization()(input_x)
    X = Conv2D(4, (5, 5), activation="tanh")(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    X = Conv2D(16, (5, 5), activation="tanh")(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    X = Flatten()(X)
    
    X = BatchNormalization()(X)
    X = Dense(NUM_CLASSES, activation="tanh")(X)
    model = Model(inputs=input_x, outputs=X, name=model_name)

    return model

def OS_CNN(model_name: str = MODELS_TYPES.ONE_SHOT_LEARNING.value) -> keras.engine.functional.Functional:
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

    model = Model(inputs=input_x, outputs=X, name=model_name)

    return model

def AlexNet_CNN(model_name: str = MODELS_TYPES.ALEX_NET.value) -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = BatchNormalization()(input_x)
    X = Conv2D(96, kernel_size=(11,11), strides=4,
                        padding='valid', activation='relu',
                        kernel_initializer='he_normal', name='conv1')(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                          padding='valid', data_format=None, name='pool1')(X)

    X = Conv2D(256, kernel_size=(5,5), strides= 1,
                    padding= 'same', activation='relu',
                    kernel_initializer='he_normal', name='conv2')(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                          padding= 'valid', data_format=None, name='pool2')(X)

    X = Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer='he_normal', name='conv3')(X)

    X = Conv2D(384, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer='he_normal', name='conv4')(X)

    X = Conv2D(256, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer='he_normal', name='conv5')(X)

    X = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                          padding= 'valid', data_format=None, name='pool5')(X)
    
    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(4096, activation= 'relu', name='fc6')(X)
    X = Dropout(0.5)(X)

    X = Dense(4096, activation= 'relu', name='fc7')(X)
    
    # May or may not be required to be removed
    X = Dense(1000, activation= 'relu', name='fc8')(X)

    X = Dense(NUM_CLASSES, activation= 'softmax', name='fc9')(X)

    model = Model(inputs=input_x, outputs=X, name=model_name)
    
    return model

def VGGNet_CNN(model_name: str = MODELS_TYPES.VGG_NET_16.value) -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = BatchNormalization()(input_x)
    X = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", name="block1_conv1")(X)
    X = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", name="block1_conv2")(X)
    X = MaxPooling2D((2, 2), name="block1_pool")(X)

    X = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name="block2_conv1")(X)
    X = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name="block2_conv2")(X)
    X = MaxPooling2D((2, 2), name="block2_pool")(X)

    X = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name="block3_conv1")(X)
    X = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name="block3_conv2")(X)
    X = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name="block3_conv3")(X)
    X = MaxPooling2D((2, 2), name="block3_pool")(X)

    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="block4_conv1")(X)
    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="block4_conv2")(X)
    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="block4_conv3")(X)
    X = MaxPooling2D((2, 2), name="block4_pool")(X)

    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="block5_conv1")(X)
    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="block5_conv2")(X)
    X = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="block5_conv3")(X)
    X = MaxPooling2D((2, 2), name="block5_pool")(X)

    X = Flatten(name="flatten")(X)
    X = Dense(4096, activation="relu", name="fc1")(X)
    X = Dense(4096, activation="relu", name="fc2")(X)

    X = Dense(NUM_CLASSES, activation="softmax")(X)

    model = Model(inputs=input_x, outputs=X, name=model_name)

    return model

def identity_block(X, f, filters, stage, block) -> keras.engine.functional.Functional:
   
    name_base = 'conv' + stage + '_block' + block
    
    F1, F2, F3 = filters

    X_shortcut = X
   
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=name_base + '_1_conv')(X)
    X = BatchNormalization(axis=3, name=name_base + '_1_bn')(X)
    X = Activation('relu', name=name_base + '_1_relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=name_base + '_2_conv')(X)
    X = BatchNormalization(axis=3, name=name_base + '_2_bn')(X)
    X = Activation('relu', name=name_base + '_2_relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=name_base + '_3_conv')(X)
    X = BatchNormalization(axis=3, name=name_base + '_3_bn')(X)

    X = Add(name=name_base + '_add')([X, X_shortcut])# SKIP Connection
    X = Activation('relu', name=name_base + '_out')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2) -> keras.engine.functional.Functional:
   
    name_base = 'conv' + stage + '_block' + block

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=name_base + '_1_conv')(X)
    X = BatchNormalization(axis=3, name=name_base + '_1_bn')(X)
    X = Activation('relu', name=name_base + '_1_relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=name_base + '_2_conv')(X)
    X = BatchNormalization(axis=3, name=name_base + '_2_bn')(X)
    X = Activation('relu', name=name_base + '_2_relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=name_base + '_3_conv')(X)
    X = BatchNormalization(axis=3, name=name_base + '_3_bn')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=name_base + '_0_conv')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=name_base + '_0_bn')(X_shortcut)

    X = Add(name=name_base + '_add')([X, X_shortcut])
    X = Activation('relu', name=name_base + '_out')(X)

    return X


def ResNet_CNN(model_name: str = MODELS_TYPES.RES_NET_50.value) -> keras.engine.functional.Functional:
    input_x = Input(INPUT_SHAPE)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1_conv')(input_x)
    X = BatchNormalization(axis=3, name='conv1_bn')(X)
    X = Activation('relu', name='conv1_relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_pool')(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage='2', block='1', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage='2', block='2')
    X = identity_block(X, 3, [64, 64, 256], stage='2', block='3')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage='3', block='1', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage='3', block='2')
    X = identity_block(X, 3, [128, 128, 512], stage='3', block='3')
    X = identity_block(X, 3, [128, 128, 512], stage='3', block='4')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage='4', block='1', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage='4', block='2')
    X = identity_block(X, 3, [256, 256, 1024], stage='4', block='3')
    X = identity_block(X, 3, [256, 256, 1024], stage='4', block='4')
    X = identity_block(X, 3, [256, 256, 1024], stage='4', block='5')
    X = identity_block(X, 3, [256, 256, 1024], stage='4', block='6')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage='5', block='1', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage='5', block='2')
    X = identity_block(X, 3, [512, 512, 2048], stage='5', block='3')

    X = GlobalAveragePooling2D(name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(256, activation='relu', name='fc1')(X)
    X = Dense(128, activation='relu', name='fc2')(X)
    
    X = Dense(NUM_CLASSES, activation='softmax', name='fc3')(X)
    
    model = Model(inputs=input_x, outputs=X, name=model_name)

    return model