import tensorflow as tf
from tensorflow import keras

def conv_block(inputs, n_filters, kernel_size):

    x = keras.layers.Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    return x

def encoder_block(inputs, n_filters, kernel_size, pool_size, dropout_probability, max_pooling=True):
   
    feature_maps = conv_block(inputs, n_filters, kernel_size)
    
    if dropout_probability > 0:
        feature_maps = keras.layers.Dropout(dropout_probability)(feature_maps)

    if max_pooling:
        next_layer_input = keras.layers.MaxPooling2D(pool_size)(feature_maps)
    else:
        next_layer_input = feature_maps

    skip_connection = feature_maps
    
    return next_layer_input, skip_connection

def decoder_block(inputs, skipped_input, n_filters, kernel_size, stride_size):

    upsampled = keras.layers.Conv2DTranspose(n_filters, kernel_size, stride_size, padding='same')(inputs)
    merged = keras.layers.Concatenate()([upsampled, skipped_input])
    
    feature_maps = conv_block(merged, n_filters, kernel_size)
    
    next_layer = feature_maps

    return next_layer


def unet_model(input_shape=(512, 512, 3), initial_filters=32, kernel_size=(3, 3), pool_size=(2, 2), up_stride_size = (2, 2), n_classes=2, dropout_probability = 0.3):

    inputs = keras.layers.Input(input_shape)

    down_block1 = encoder_block(inputs, initial_filters, kernel_size, pool_size, 0)
    down_block2 = encoder_block(down_block1[0], initial_filters*2, kernel_size, pool_size, 0)
    down_block3 = encoder_block(down_block2[0], initial_filters*4, kernel_size, pool_size, dropout_probability)
    down_block4 = encoder_block(down_block3[0], initial_filters*8, kernel_size, pool_size, dropout_probability)

    bottleneck_block = encoder_block(down_block4[0], initial_filters*16, kernel_size, pool_size, dropout_probability, max_pooling=False)

    up_block4 = decoder_block(bottleneck_block[0], down_block4[1], initial_filters*8, kernel_size, up_stride_size)
    up_block3 = decoder_block(up_block4, down_block3[1], initial_filters*4, kernel_size, up_stride_size)
    up_block2 = decoder_block(up_block3, down_block2[1], initial_filters*2, kernel_size, up_stride_size)
    up_block1 = decoder_block(up_block2, down_block1[1], initial_filters, kernel_size, up_stride_size)

    outputs = keras.layers.Conv2D(n_classes, 1, padding='same', kernel_initializer='he_normal')(up_block1)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
