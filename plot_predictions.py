import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from patchify import (patchify, unpatchify)
from utils import center_crop, rebuild_from_patches

# Load the input and ground truth images 
x_initial_valid = plt.imread('validation/201-INPUT.jpg')
y_initial_valid = plt.imread('validation/201-OUTPUT-GT.png')

CROP_SIZE_W = 2560 
CROP_SIZE_H = 2560

x_initial_valid = center_crop(x_initial_valid, (CROP_SIZE_H, CROP_SIZE_W))
y_initial_valid = center_crop(y_initial_valid, (CROP_SIZE_H, CROP_SIZE_W))

# Display a list with the available models and ask the user to choose which to use
models_path = 'trained_models/cropped_001'
models_list = os.listdir(models_path)

print('-----------------------------------------------------')
models = os.listdir(models_path)
model = keras.models.load_model(os.path.join(models_path, models[0]))
c = 0
"""
for m in models:
    model = keras.models.load_model(os.path.join(models_path, m))
    #if c == 0:  os.system('clear')
    print(f'{m} is loaded.')
    c += 
"""

    
    

