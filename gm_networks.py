import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import center_crop

def conv_block(inputs, n_filters, kernel_size, padding):
    """
    Convolutional block of Unet, consisted of 2 sub-blocks of convolution/batch normalization/activation operations

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        kernel_size -- Size of the kernel for both convolutions
    Returns:
        x -- Tensor with the feature maps after the convolutions
    """
    x = keras.layers.Conv2D(n_filters, kernel_size, padding=padding, kernel_initializer='he_normal')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(n_filters, kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    return x

def encoder_block(inputs, n_filters, kernel_size, padding, pool_size, dropout_probability, max_pooling=True):
    """ 
    A typical block of the encoder of Unet, including a convolutional block, the dropout probability and the max pooling to reduce the spatial dimensions

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        kernel_size -- Size of the kernel for the convolutional block
        pool_size -- Size of the kernel for the max pooling operation
        dropout_probability -- Dropout probability
        max_pooling -- Boolean for the use of max pooling operation
    Returns:
        next_layer -- Output to the next block
        skip_connection --  Output for the skip connection
    """
    feature_maps = conv_block(inputs, n_filters, kernel_size, padding)
    
    if dropout_probability > 0:
        feature_maps = keras.layers.Dropout(dropout_probability)(feature_maps)

    if max_pooling:
        next_layer_input = keras.layers.MaxPooling2D(pool_size)(feature_maps)
    else:
        next_layer_input = feature_maps

    skip_connection = feature_maps
    
    return next_layer_input, skip_connection

def decoder_block(inputs, skipped_input, n_filters, kernel_size, padding, stride_size):
    """
    A typical block of the decoder of Unet, including a transposed convolution, concatenate and a convolutional block

    Arguments:
        inputs -- Input tensor 
        skipped_input -- Input tensor from a previous skip coneection
        n_filters -- Number of filters for the convolutional layers
        kernel_size -- Size of the kernel for the convolutional block
        stride_size -- Stride size for the trasposed convolution
    Returns:
        next_layer -- Output to the next block
    """
    upsampled = keras.layers.Conv2DTranspose(n_filters, kernel_size, stride_size, padding='same')(inputs)
    merged = keras.layers.Concatenate()([upsampled, skipped_input])
    feature_maps = conv_block(merged, n_filters, kernel_size, padding)
    next_layer = feature_maps
    
    return next_layer


def unet_model(input_shape=(512, 512, 3), initial_filters=32, kernel_size=(3, 3), padding='same', pool_size=(2, 2), up_stride_size = (2, 2), n_classes=1, dropout_probability = 0.3):
    """
    Definition of the Unet model

    Arguments:
        input_shape -- Shape of the input images to the model
        initial_filters -- Number of the filters of the first convolutional block
        kernel_size -- Size of the kernel for the convolutional block
        pool_size -- Size of the kernel for the max pooling operation
        up_stride_size -- Stride size for the trasposed convolution
        n_classes -- Number of the output classes
        dropout_probability -- Dropout probability
    Returns:
        The model
    """
    inputs = keras.layers.Input(input_shape)
    down_block1 = encoder_block(inputs, initial_filters, kernel_size, padding, pool_size, 0)
    down_block2 = encoder_block(down_block1[0], initial_filters*2, kernel_size, padding, pool_size, 0)
    down_block3 = encoder_block(down_block2[0], initial_filters*4, kernel_size, padding, pool_size, 0)
    down_block4 = encoder_block(down_block3[0], initial_filters*8, kernel_size, padding, pool_size, dropout_probability)
    bottleneck_block = encoder_block(down_block4[0], initial_filters*16, kernel_size, padding, pool_size, dropout_probability, max_pooling=False)
    up_block4 = decoder_block(bottleneck_block[0], down_block4[1], initial_filters*8, kernel_size, padding, up_stride_size)
    up_block3 = decoder_block(up_block4, down_block3[1], initial_filters*4, kernel_size, padding, up_stride_size)
    up_block2 = decoder_block(up_block3, down_block2[1], initial_filters*2, kernel_size, padding, up_stride_size)
    up_block1 = decoder_block(up_block2, down_block1[1], initial_filters, kernel_size, padding, up_stride_size)
    outputs = keras.layers.Conv2D(n_classes, 1, padding='same', activation='sigmoid', kernel_initializer='he_normal')(up_block1)

    return keras.Model(inputs=inputs, outputs=outputs)
