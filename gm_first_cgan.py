import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Disables the INFO messages

from tensorflow import keras
import numpy as np
from gm_networks import unet_model
from matplotlib import pyplot as plt

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

def compile_discriminator(model, optimizer=keras.optimizers.Adam(), 
	losses='binary_crossentropy'):

	model.compile(optimizer=optimizer, loss=losses)

def build_generator():
	return unet_model()

def build_gan(generator, discriminator, input_shape=None):

	discriminator.trainable = False

	inputs = keras.layers.Input(shape=input_shape)
	generated_images = generator(inputs)
	discriminator_outputs = discriminator([inputs, generated_images])
	
	return keras.Model(inputs, [discriminator_outputs, generated_images])

def compile_gan(model, optimizer=keras.optimizers.Adam(), 
	losses=['binary_crossentropy', 'mae'], losses_weights=[1, 100], 
	metrics=['accuracy']):
	
	model.compile(optimizer=optimizer, loss=losses, loss_weights=losses_weights,
	metrics=metrics)

def train(batches, generator, discriminator, gan, batch_size=None,
	epochs=None, steps_per_epoch=None, impair_for_predictions=None,
	path_to_save_predictions=None):
	#===========================================================================
	"""NOTE. Be very carefull here!!! "x_batch" and "y_batch" are Numpy
		Iterators. They iterate every time they are called. So, if they are not 
		called equal times for every process in the function, they stop being 
		consistent, i.e. they do not refer to a valid pair of input-target images
	"""
	x_batch, y_batch = batches
	#===========================================================================
	valid_labels = np.ones((batch_size, 1))
	fake_labels = np.zeros((batch_size, 1))
	g_loss_history = []
	d_loss_real_history = []
	d_loss_fake_history = []
	for epoch in range(epochs):
		for step in range(steps_per_epoch):
			x_real = x_batch[step]
			y_real = y_batch[step]
			y_fake = generator.predict(x_real, verbose=0)
			d_loss_real = discriminator.train_on_batch([x_real, y_real], 
				valid_labels)
			d_loss_fake = discriminator.train_on_batch([x_real, y_fake], 
				fake_labels)
			g_loss, _, _, _, _ = gan.train_on_batch(x_real, [valid_labels, 
				y_real])

		if epoch == 0 or (epoch + 1) % 10 == 0:
			input_to_predict = impair_for_predictions[0]
			ground_truth = impair_for_predictions[1]
			predidtion = np.squeeze(generator.predict(input_to_predict, 
				verbose=0))

			fig, ax = plt.subplots(1, 3)
			ax[0].imshow(np.squeeze(input_to_predict/255))
			ax[0].set_title(f'Input')
			ax[0].set_xticks([])
			ax[0].set_yticks([])
			ax[1].imshow(ground_truth, cmap='Greys_r')
			ax[1].set_title('Ground Truth')
			ax[1].set_xticks([])
			ax[1].set_yticks([])
			ax[2].imshow(np.where(predidtion > 0.5, 1, 0), cmap='Greys_r')
			ax[2].set_title(f'Epoch = {epoch+1}')
			ax[2].set_xticks([])
			ax[2].set_yticks([])

			pathname = os.path.join(path_to_save_predictions, 
				f'epoch_{epoch+1}.png')
			fig.savefig(pathname, dpi=250)
			plt.close()
		
		print(f'Epoch {epoch+1}, g_loss = {"{:0.4f}".format(g_loss)}, '
			f'd_loss_real = {"{:0.4f}".format(d_loss_real)}, '
			f'd_loss_fake = {"{:0.4f}".format(d_loss_fake)}')

		g_loss_history.append(g_loss)
		d_loss_real_history.append(d_loss_real)
		d_loss_fake_history.append(d_loss_fake)
	
	history = {
		'g_loss': g_loss_history,
		'd_loss_real': d_loss_real_history,
		'd_loss_fake': d_loss_fake_history 
	}
	return history

	
