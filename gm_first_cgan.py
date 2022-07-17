# example of defining a 70x70 patchgan discriminator model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Disables the INFO messages
from tensorflow import keras
from gm_networks import unet_model

def build_discriminator(input_images_shape=(512, 512, 3), 
	target_images_shape=(512, 512, 1), d_layers=4, initial_filters=64, 
	kernel_size=3, strides=2, padding='same', dropout_probability=0.3):

	input_A = keras.Input(shape=input_images_shape)
	input_B = keras.Input(shape=target_images_shape)
	x = keras.layers.Concatenate()([input_A, input_B])
	multiplier=1
	for i in range(d_layers):
		n_filters = initial_filters*multiplier
		x = keras.layers.Conv2D(n_filters, kernel_size, 
			strides=strides, 
			padding=padding, kernel_initializer='he_normal', 
			name='conv2D_' + str(i))(x)

		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Activation('relu')(x)

		if dropout_probability > 0:
			x = keras.layers.Dropout(rate=dropout_probability)(x)
		
		if n_filters == 256:
			multiplier = multiplier
		else:
			multiplier *= 2
		
	x = keras.layers.Flatten()(x)
	d_output = keras.layers.Dense(1, activation='sigmoid')(x) 

	return keras.Model([input_A, input_B], d_output)

def compile_discriminator(model, optimizer=None, losses='binary_crossentropy', 
	metrics=['Accuracy']):

	model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

def train_discriminator():
	pass

def build_generator():
	return unet_model()

def build_gan(generator, discriminator, input_images_shape):

	discriminator.trainable = False

	inputs = keras.layers.Input(shape=input_images_shape)
	generated_images = generator(inputs)
	discriminator_outputs = discriminator([inputs, generated_images])
	
	return keras.Model(inputs, [discriminator_outputs, generated_images])

def compile_gan(model, optimizer=None, losses=['binary_crossentropy', 
	'mae'], losses_weights=[1, 100], metrics=['accuracy']):
	
	model.compile(optimizer=optimizer, loss=losses, loss_weights=losses_weights,
	metrics=metrics)

def train(generator, discriminator, ):
	pass