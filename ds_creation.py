#TODO: rewrite everything using the argparse module, so that I can do everything
#from the terminal
from ast import arg
import os
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
import tensorflow as tf
from patchify import patchify
from tkinter.filedialog import askopenfilename, askdirectory
from gui_for_ds import get_arguments_by_gui

# NOTE: The following line sets the tensorflow's devive to CPU. The reason is 
# that the TensorFlow function used in this script is used in a for loop.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def create_patches(input_image, target_image, patch_size=128, 
    sampling_method=None, patches_number=None, patches_step=None):
    
    input_image = np.expand_dims(input_image, 0)
    target_image = np.expand_dims(target_image, (0, -1))
    concatenated_images = np.concatenate([input_image, target_image], -1)

    if sampling_method == 'independent_patches':  
        patches = patchify(concatenated_images, (1, patch_size, patch_size, concatenated_images.shape[-1]), step=patch_size)
        new_images = np.reshape(patches, ((-1,) + patches.shape[-3:]))
        
    elif sampling_method == 'overlapped_patches':
        patches = patchify(concatenated_images, (1, patch_size, patch_size, concatenated_images.shape[-1]), 
            step=patches_step)
        new_images = np.reshape(patches, ((-1,) + patches.shape[-3:]))
        
    elif sampling_method == 'random_patches':
        new_images = np.zeros([patches_number, patch_size, patch_size, 4])

        for i in range(patches_number):
            t = tf.image.random_crop(concatenated_images, size=[1, patch_size, patch_size, 4])
            new_images[i] = t[0]
            
    else:
        raise ValueError('Invalid value was given for the arument "mode".')
    
    print('Patches created: ', new_images.shape[0])

    return new_images

def remove_invalid_patches(images):
    # Add all the channels values pixel-wise 
    added_channels = np.sum(images, -1)
    # The pixels that result to 0 correspond to areas that are outside the map borders
    invalid_pixels_id = np.where(added_channels == 0, 0, 1)
    # Find the patches that contain only valid pixels
    valid_patches_id = invalid_pixels_id.all(axis=(1, 2))
    # Keep only the valid patches
    valid_patches = images[valid_patches_id]
    print('Valid patches (without areas that exceed the map borders): ',
        valid_patches.shape[0])

    return valid_patches


def remove_non_texture_patches(images, min_class_percentage=None):

    max_class_percentage = 1 - min_class_percentage
    target_images = images[:, :, :, -1]
    img_pixels_num = target_images.shape[1] * target_images.shape[2]
    percentages = np.count_nonzero(target_images, axis=(1, 2)) / img_pixels_num
    patches_w_texture_id = np.where((percentages < min_class_percentage) | (percentages > max_class_percentage), False, True)
    patches_w_texture = images[patches_w_texture_id]
    print('Patches with texture: ', patches_w_texture.shape[0])
    return patches_w_texture


def save_dataset(input_images, target_images, output_directory=None, 
    sampling_method=None, patch_size=None, initial_ds_name=None):
    error_msg = ['The number of the input images, as long as the number of the ' 
    'target images must be the same. You must check the provided arguments in '
    'the function']

    target_images = target_images * 255
    if len(target_images.shape) == 3: # if the target image is grayscale
        assert input_images.shape[:-1] == target_images.shape, error_msg[0]
        target_images = np.expand_dims(target_images, axis=-1)
        # Repeat the tensor with the binary images (with shape = [images_num, height, width, 1]) 3 times, so as to save it as a RGB image 
        target_images = np.tile(target_images, (1, 1, 1, 3)).astype(np.uint8)
    
    else: # if the target image is multi-channel
        assert input_images.shape[:-1] == target_images.shape[-1], error_msg[0]

    patches_sz_level = f'{patch_size}by{2*patch_size}'
    ds_final_dir_level = os.path.join(output_directory, patches_sz_level, 
        sampling_method, initial_ds_name)
    
    images_number = input_images.shape[0]
    if sampling_method == 'random_patches':

        ds_final_dir_level = os.path.join(ds_final_dir_level, 
            str(images_number) + 'ims')
        
    os.makedirs(ds_final_dir_level)
    target_images = np.where(target_images == 1, 255, 0).astype(np.uint8)
    
    for i in range(input_images.shape[0]):
        images = np.hstack((input_images[i], target_images[i]))
        filename = f'{i+1}.jpg'
        filepath = os.path.join(ds_final_dir_level, filename)
        im = Image.fromarray(images)
        im.save(filepath)

    print(f'\nThe final patches (= {images_number}) were saved in : '
       f'{ds_final_dir_level}')


if __name__ == '__main__':

    args = get_arguments_by_gui()
    input_image_path = args['input image path']
    target_image_path = args['target image path']
    out_dir = args['output directory']
    method = args['sampling method']
    patch_sz = args['patch size']
    min_perc = args['minimum percentage']

    if 'patches number' in args:
        patch_num = args['patches number']
    else:
        patch_num = None

    if 'patches step' in args:
        patch_stp = args['patches step']
    else:
        patch_stp = None
       
    input_image = Image.open(input_image_path)
    target_image = Image.open(target_image_path)
    print('\n\33[4mDataset creation report \33[m\n')
    new_patches = create_patches(input_image, target_image, patch_size=patch_sz, 
        sampling_method=method, patches_number=patch_num, patches_step=patch_stp)
    
    valid_patches = remove_invalid_patches(new_patches)
    final_patches = remove_non_texture_patches(valid_patches, 
        min_class_percentage=min_perc)

    # NOTE: The ds_name in the following 2 lines is extracted, according to the 
    # current dataset's naming conventions!!! It would not work in every case
    input_image_name = input_image_path.split('/')[-1]
    ds_name = input_image_name.split('-')[0]
    ds_input_images = final_patches[:, :, :, :-1].astype(np.uint8)
    ds_target_images = final_patches[:, :, :, -1]
    save_dataset(ds_input_images, ds_target_images, output_directory=out_dir, 
        sampling_method=method, patch_size=patch_sz, initial_ds_name=ds_name)

    


    


