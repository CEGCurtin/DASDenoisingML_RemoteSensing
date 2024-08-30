# -*- coding: utf-8 -*-
"""
Neural Network for DAS Noise2Noise

Created on Thu May 11 11:09:50 2023

@author: 253863J
"""
from tensorflow import keras

def UNet(input_size = (128,96), filters = [24, 24, 48, 48]):
    """
    Build U-net architecture

    Parameters
    ----------
    input_size : tuple (n_rows,n_col), optional
        Input size to the neural netowrk. The default is (128,96).
    filters : list, optional
        Filters. The default is [24, 24, 48, 48].

    Returns
    -------
    model : keras.engine.functional.Functional
        UNet model.
        
    Notes
    -----
    Adapted from https://www.machinelearningnuggets.com/image-segmentation-with-u-net-define-u-net-model-from-scratch-in-keras-and-tensorflow/

    """

    input_size = input_size
    num_classes = 1
    f = filters

    inputs = keras.layers.Input(input_size)
    c0 = keras.layers.Reshape((input_size[0], input_size[1], 1))(inputs)

    #Contraction path
    c1 = keras.layers.Conv2D(f[0], (3, 3), padding='same')(c0)
    r1 = keras.layers.LeakyReLU(alpha=0.1)(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(r1)

    c2 = keras.layers.Conv2D(f[1], (3, 3), padding='same')(p1)
    r2 = keras.layers.LeakyReLU(alpha=0.1)(c2)

    #Expansive path 
    u1 = keras.layers.UpSampling2D((2,2))(r2)

    u2 = keras.layers.concatenate([u1, c1])
 
    u3 = keras.layers.Conv2D(f[2], (3, 3), padding='same')(u2)
    r3 = keras.layers.LeakyReLU(alpha=0.1)(u3)
 
    u4 = keras.layers.Conv2D(f[3], (3, 3), padding='same')(r3)
    r4 = keras.layers.LeakyReLU(alpha=0.1)(u4)
 
    u5 = keras.layers.Conv2D(num_classes, (1, 1), activation='linear')(r4)
    
    outputs = keras.layers.Reshape((input_size[0], input_size[1]))(u5)
    model = keras.models.Model(inputs, outputs)
    
    return model

def FCDNet(input_size = (128,128), filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]):
    """
    Build FCDNet architecture

    Parameters
    ----------
    input_size : tuple (n_rows,n_col), optional
        Input size to the neural netowrk. The default is (128,96).
    filters : list, optional
        Filters. The default is [24, 24, 48, 48].

    Returns
    -------
    model : keras.engine.functional.Functional
        UNet model.
        
    Notes
    -----
    Adapted from https://www.machinelearningnuggets.com/image-segmentation-with-u-net-define-u-net-model-from-scratch-in-keras-and-tensorflow/
    Initially coded by Xihao (neural_FCDnet)

    """

    input_size = input_size
    num_classes = 1
    f = filters

    inputs = keras.layers.Input(input_size)
    c0 = keras.layers.Reshape((input_size[0], input_size[1], 1))(inputs)

    #Contraction path
    c1 = keras.layers.Conv2D(32, (3, 3), padding='same')(c0)
    c1 = keras.layers.ReLU()(c1)
    c1 = keras.layers.BatchNormalization()(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = keras.layers.Conv2D(64, (3, 3), padding='same')(p1)
    c2 = keras.layers.ReLU()(c2)
    c2 = keras.layers.BatchNormalization()(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = keras.layers.Conv2D(128, (3, 3), padding='same')(p2)
    c3 = keras.layers.ReLU()(c3)
    c3 = keras.layers.BatchNormalization()(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = keras.layers.Conv2D(256, (3, 3), padding='same')(p3)
    c4 = keras.layers.ReLU()(c4)
    c4 = keras.layers.BatchNormalization()(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    
    c5 = keras.layers.Conv2D(512, (3, 3), padding='same')(p4)
    c5 = keras.layers.ReLU()(c5)
    c5 = keras.layers.BatchNormalization()(c5)
    c5 = keras.layers.Dropout(0.2)(c5)
    
    #Convolutional block
    cb5 = keras.layers.Conv2D(512, (3, 3), padding='same')(c5)
    cb5 = keras.layers.ReLU()(cb5)
    cb5 = keras.layers.BatchNormalization()(cb5)
    cb5 = keras.layers.Dropout(0.2)(cb5)
    
    cb4 = keras.layers.Conv2D(256, (3, 3), padding='same')(c4)
    cb4 = keras.layers.ReLU()(cb4)
    cb4 = keras.layers.BatchNormalization()(cb4)
    cb4 = keras.layers.Dropout(0.2)(cb4)
    
    cb3 = keras.layers.Conv2D(128, (3, 3), padding='same')(c3)
    cb3 = keras.layers.ReLU()(cb3)
    cb3 = keras.layers.BatchNormalization()(cb3)
    cb3 = keras.layers.Dropout(0.2)(cb3)
    
    cb2 = keras.layers.Conv2D(64, (3, 3), padding='same')(c2)
    cb2 = keras.layers.ReLU()(cb2)
    cb2 = keras.layers.BatchNormalization()(cb2)
    cb2 = keras.layers.Dropout(0.2)(cb2)
    
    cb1 = keras.layers.Conv2D(32, (3, 3), padding='same')(c1)
    cb1 = keras.layers.ReLU()(cb1)
    cb1 = keras.layers.BatchNormalization()(cb1)
    cb1 = keras.layers.Dropout(0.2)(cb1)
    
    #Expansive path 
    u6 = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(cb5)
    u6 = keras.layers.concatenate([u6, cb4])
    u6 = keras.layers.Conv2D(256, (3, 3), padding='same')(u6)   
    u6 = keras.layers.ReLU()(u6)
    u6 = keras.layers.BatchNormalization()(u6)
    
    u7 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = keras.layers.concatenate([u7, cb3])
    u7 = keras.layers.Conv2D(128, (3, 3), padding='same')(u7)   
    u7 = keras.layers.ReLU()(u7)
    u7 = keras.layers.BatchNormalization()(u7)
    
    u8 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = keras.layers.concatenate([u8, cb2])
    u8 = keras.layers.Conv2D(64, (3, 3), padding='same')(u8)   
    u8 = keras.layers.ReLU()(u8)
    u8 = keras.layers.BatchNormalization()(u8)
    
    u9 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = keras.layers.concatenate([u9, cb1])
    u9 = keras.layers.Conv2D(32, (3, 3), padding='same')(u9)   
    u9 = keras.layers.ReLU()(u9)
    u9 = keras.layers.BatchNormalization()(u9)
    
    u10=keras.layers.Conv2D(num_classes, (1, 1), activation='linear')(u9)   
    outputs = keras.layers.Reshape((input_size[0], input_size[1]))(u10)
    model = keras.models.Model(inputs, outputs)
    
    return model