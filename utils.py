import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify

def normalize_data(data):
    
    return (data - data.min()) / (data.max() - data.min())

def center_crop(image, crop_shape):
   
    crop_rows, crop_cols = crop_shape[:2]

    if len(image.shape) == 2:
        image_rows, image_cols = image.shape

        start_row = int((image_rows - crop_rows) / 2)    
        start_col = int((image_cols - crop_cols) / 2)
        end_row = start_row + crop_rows
        end_col = start_col + crop_cols

        cropped_image = image[start_row:end_row, start_col:end_col]

    elif len(image.shape) == 3:
        image_rows, image_cols, _ = image.shape

        start_row = int((image_rows - crop_rows) / 2)
        start_col = int((image_cols - crop_cols) / 2)
        end_row = start_row + crop_rows
        end_col = start_col + crop_cols

        cropped_image = image[start_row:end_row, start_col:end_col, :]

    else:
        _, image_rows, image_cols, _ = image.shape
    
        start_row = int((image_rows - crop_rows) / 2)    
        start_col = int((image_cols - crop_cols) / 2)
        end_row = start_row + crop_rows
        end_col = start_col + crop_cols

        cropped_image = image[:, start_row:end_row, start_col:end_col, :]

    return cropped_image 


def create_dataset_from_one_image(input_image, target_image, new_image_size, mode='independent_patches', 
                                  new_images_number=None, patches_step=None):
        
    input_image = np.expand_dims(input_image, -1) if len(input_image.shape) == 2 else input_image
    target_image = np.expand_dims(target_image, -1) if len(target_image.shape) == 2 else target_image
    
    concatenated_images = np.concatenate([input_image, target_image], -1)

    if mode == 'independent_patches':
       
        patches = patchify(concatenated_images, (new_image_size, new_image_size, concatenated_images.shape[-1]), 
                           step=new_image_size)
        new_images = np.reshape(patches, ((-1,) + patches.shape[-3:]))
        
    elif mode == 'overlapped_patches':
        
        if not isinstance(patches_step, int) or patches_step<0:
            raise ValueError(f'A positive integer was expected for the argument "patches_step".' )
        
        patches = patchify(concatenated_images, (new_image_size, new_image_size, concatenated_images.shape[-1]), 
                           step=patches_step)
        new_images = np.reshape(patches, ((-1,) + patches.shape[-3:]))
        
    elif mode == 'random_patches':
        
        if not isinstance(new_images_number, int) or new_images_number<0:
            raise ValueError(f'A positive integer was expected for the argument "new_images_number".' )
            
        new_images = np.zeros([new_images_number, new_image_size, new_image_size, concatenated_images.shape[-1]])
        
        for i in range(new_images_number):

            t = tf.image.random_crop(concatenated_images, size=[new_image_size, new_image_size, concatenated_images.shape[-1]])
            new_images[i] = t[0]
            
    else:
        raise ValueError('Invalid value was given for the arument "mode".')
    
    # Separate the inputs from the outputs to create the final batches
    input_dataset = new_images[:, :, :, :-1]
    target_dataset = np.expand_dims(new_images[:, :, :, -1], -1)

    return input_dataset, target_dataset


def rebuild_from_patches(predictions, initial_image_shape, patch_step, patch_side):

    overlap_ratio = patch_side / patch_step
    assert overlap_ratio in (2, 4), "The patch step can only be 1/2 or 1/4 of the patch side" 

    patch_side = int(patch_side)
    half_patch_side = int(patch_side / 2)

    patches_num_vertically = predictions.shape[0]
    patches_num_horizontally = predictions.shape[1]

    if overlap_ratio == 2:

        unified_predictions = np.empty(initial_image_shape + (4, ))
        unified_predictions[:] = np.NaN

        for row in range(patches_num_vertically):
            for col in range(patches_num_horizontally):
                
                start_row = patch_step * row
                middle_row = start_row + half_patch_side
                end_row = start_row + patch_side

                start_col = patch_step * col
                middle_col = start_col + half_patch_side
                end_col = start_col + patch_side

                if row == 0 and col == 0:
                    unified_predictions[:patch_side, :patch_side, 0] = predictions[row, col]
                
                elif row == 0:
                    unified_predictions[start_row:end_row, start_col:middle_col, 1] = predictions[row, col, :, :half_patch_side]
                    unified_predictions[start_row:end_row, middle_col:end_col, 0] = predictions[row, col, :, half_patch_side:]

                elif col == 0:
                    unified_predictions[start_row:middle_row, start_col:middle_col, 1] = predictions[row, col, :half_patch_side, :half_patch_side]
                    unified_predictions[start_row:middle_row, middle_col:end_col, 2] = predictions[row, col, :half_patch_side, half_patch_side:]
                    unified_predictions[middle_row:end_row, start_col:end_col, 0] = predictions[row, col, half_patch_side:, :]

                else:
                    unified_predictions[start_row:middle_row, start_col:middle_col, 3] = predictions[row, col, :half_patch_side, :half_patch_side]
                    unified_predictions[start_row:middle_row, middle_col:end_col, 2] = predictions[row, col, :half_patch_side, half_patch_side:]
                    unified_predictions[middle_row:end_row, start_col:middle_col, 1] = predictions[row, col, half_patch_side:, :half_patch_side]
                    unified_predictions[middle_row:end_row, middle_col:end_col, 0] = predictions[row, col, half_patch_side:, half_patch_side:]

    elif overlap_ratio == 4:

        unified_predictions = np.empty(initial_image_shape + (16, ))
        unified_predictions[:] = np.NaN

        quarter_patch_side = int(patch_side / 4)
        three_quarters_patch_side = int(quarter_patch_side * 3)

        for row in range(patches_num_vertically):
            for col in range(patches_num_horizontally):

                start_row = patch_step * row
                quarter_row = start_row + quarter_patch_side
                middle_row = start_row + half_patch_side
                three_quarters_row = start_row + three_quarters_patch_side
                end_row = start_row + patch_side

                start_col = patch_step * col
                quarter_col = start_col + quarter_patch_side
                middle_col = start_col + half_patch_side
                three_quarters_col = start_col + three_quarters_patch_side
                end_col = start_col + patch_side

                if row == 0 and col == 0:
                    unified_predictions[:patch_side, :patch_side, 0] = predictions[row, col]
                    
                elif row == 0 and col == 1:
                    unified_predictions[start_row:end_row, start_col:three_quarters_col, 1] = predictions[row, col, :, :three_quarters_patch_side]
                    unified_predictions[start_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, :, three_quarters_patch_side:]
                    
                elif row == 0 and col == 2:
                    unified_predictions[start_row:end_row, start_col:middle_col, 2] = predictions[row, col, :, :half_patch_side]
                    unified_predictions[start_row:end_row, middle_col:three_quarters_col, 1] = predictions[row, col, :, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, :, three_quarters_patch_side:]
                    
                elif row == 0 and col > 2:
                    unified_predictions[start_row:end_row, start_col:quarter_col, 3] = predictions[row, col, :, :quarter_patch_side]
                    unified_predictions[start_row:end_row, quarter_col:middle_col, 2] = predictions[row, col, :, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:end_row, middle_col:three_quarters_col, 1] = predictions[row, col, :, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, :, three_quarters_patch_side:]
                    
                elif row == 1 and col == 0:
                    unified_predictions[start_row:three_quarters_row, start_col:quarter_col, 1] = predictions[row, col, :three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:three_quarters_row, quarter_col:middle_col, 2] = predictions[row, col, :three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:three_quarters_row, middle_col:three_quarters_col, 3] = predictions[row, col, :three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, :three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, :]
                                    
                elif row == 1 and col == 1:
                    unified_predictions[start_row:three_quarters_row, start_col:quarter_col, 3] = predictions[row, col, :three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:three_quarters_row, quarter_col:middle_col, 4] = predictions[row, col, :three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, :three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, :three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, :three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]     
                    
                elif row == 1 and col == 2:
                    unified_predictions[start_row:three_quarters_row, start_col:quarter_col, 5] = predictions[row, col, :three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:three_quarters_row, quarter_col:middle_col, 6] = predictions[row, col, :three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, :three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, :three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:middle_col, 2] = predictions[row, col, three_quarters_patch_side:, :half_patch_side]
                    unified_predictions[three_quarters_row:end_row, middle_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, half_patch_side:three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]
                                        
                elif row == 1 and col > 2:

                    unified_predictions[start_row:three_quarters_row, start_col:quarter_col, 7] = predictions[row, col, :three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:three_quarters_row, quarter_col:middle_col, 6] = predictions[row, col, :three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, :three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, :three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:quarter_col, 3] = predictions[row, col, three_quarters_patch_side:, :quarter_patch_side]
                    unified_predictions[three_quarters_row:end_row, quarter_col:middle_col, 2] = predictions[row, col, three_quarters_patch_side:, quarter_patch_side:half_patch_side]
                    unified_predictions[three_quarters_row:end_row, middle_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, half_patch_side:three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]
                                        
                elif row == 2 and col == 0:

                    unified_predictions[start_row:middle_row, start_col:quarter_col, 2] = predictions[row, col, :half_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:middle_row, quarter_col:middle_col, 4] = predictions[row, col, :half_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:middle_row, middle_col:three_quarters_col, 6] = predictions[row, col, :half_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:middle_row, three_quarters_col:end_col, 8] = predictions[row, col, :half_patch_side, three_quarters_patch_side:]
                    unified_predictions[middle_row:three_quarters_row, start_col:quarter_col, 1] = predictions[row, col, half_patch_side:three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[middle_row:three_quarters_row, quarter_col:middle_col, 2] = predictions[row, col, half_patch_side:three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[middle_row:three_quarters_row, middle_col:three_quarters_col, 3] = predictions[row, col, half_patch_side:three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[middle_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, :]
                
                elif row == 2 and col == 1:
                    unified_predictions[start_row:middle_row, start_col:quarter_col, 5] = predictions[row, col, :half_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:middle_row, quarter_col:middle_col, 7] = predictions[row, col, :half_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:middle_row, middle_col:three_quarters_col, 9] = predictions[row, col, :half_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:middle_row, three_quarters_col:end_col, 8] = predictions[row, col, :half_patch_side, three_quarters_patch_side:]
                    unified_predictions[middle_row:three_quarters_row, start_col:quarter_col, 3] = predictions[row, col, half_patch_side:three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[middle_row:three_quarters_row, quarter_col:middle_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[middle_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, half_patch_side:three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[middle_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, :three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]
                    
                elif row == 2 and col == 2:
                    unified_predictions[start_row:middle_row, start_col:quarter_col, 8] = predictions[row, col, :half_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:middle_row, quarter_col:middle_col, 10] = predictions[row, col, :half_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:middle_row, middle_col:three_quarters_col, 9] = predictions[row, col, :half_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:middle_row, three_quarters_col:end_col, 8] = predictions[row, col, :half_patch_side, three_quarters_patch_side:]
                    unified_predictions[middle_row:three_quarters_row, start_col:quarter_col, 5] = predictions[row, col, half_patch_side:three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[middle_row:three_quarters_row, quarter_col:middle_col, 6] = predictions[row, col, half_patch_side:three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[middle_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, half_patch_side:three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[middle_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:middle_col, 2] = predictions[row, col, three_quarters_patch_side:, :half_patch_side]
                    unified_predictions[three_quarters_row:end_row, middle_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, half_patch_side:three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]
                                    
                elif row == 2 and col > 2:

                    unified_predictions[start_row:middle_row, start_col:quarter_col, 11] = predictions[row, col, :half_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:middle_row, quarter_col:middle_col, 10] = predictions[row, col, :half_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:middle_row, middle_col:three_quarters_col, 9] = predictions[row, col, :half_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:middle_row, three_quarters_col:end_col, 8] = predictions[row, col, :half_patch_side, three_quarters_patch_side:]
                    unified_predictions[middle_row:three_quarters_row, start_col:quarter_col, 7] = predictions[row, col, half_patch_side:three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[middle_row:three_quarters_row, quarter_col:middle_col, 6] = predictions[row, col, half_patch_side:three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[middle_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, half_patch_side:three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[middle_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:quarter_col, 3] = predictions[row, col, three_quarters_patch_side:, :quarter_patch_side]
                    unified_predictions[three_quarters_row:end_row, quarter_col:middle_col, 2] = predictions[row, col, three_quarters_patch_side:, quarter_patch_side:half_patch_side]
                    unified_predictions[three_quarters_row:end_row, middle_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, half_patch_side:three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]
                
                elif row > 2 and col == 0:
                    unified_predictions[start_row:quarter_row, start_col:quarter_col, 3] = predictions[row, col, :quarter_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:quarter_row, quarter_col:middle_col, 6] = predictions[row, col, :quarter_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:quarter_row, middle_col:three_quarters_col, 9] = predictions[row, col, :quarter_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:quarter_row, three_quarters_col:end_col, 12] = predictions[row, col, :quarter_patch_side, three_quarters_patch_side:]
                    unified_predictions[quarter_row:middle_row, start_col:quarter_col, 2] = predictions[row, col, quarter_patch_side:half_patch_side, :quarter_patch_side]
                    unified_predictions[quarter_row:middle_row, quarter_col:middle_col, 4] = predictions[row, col, quarter_patch_side:half_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[quarter_row:middle_row, middle_col:three_quarters_col, 6] = predictions[row, col, quarter_patch_side:half_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[quarter_row:middle_row, three_quarters_col:end_col, 8] = predictions[row, col, quarter_patch_side:half_patch_side, three_quarters_patch_side:]
                    unified_predictions[middle_row:three_quarters_row, start_col:quarter_col, 1] = predictions[row, col, half_patch_side:three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[middle_row:three_quarters_row, quarter_col:middle_col, 2] = predictions[row, col, half_patch_side:three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[middle_row:three_quarters_row, middle_col:three_quarters_col, 3] = predictions[row, col, half_patch_side:three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[middle_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, :]
                                   
                elif row > 2 and col == 1:

                    unified_predictions[start_row:quarter_row, start_col:quarter_col, 7] = predictions[row, col, :quarter_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:quarter_row, quarter_col:middle_col, 10] = predictions[row, col, :quarter_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:quarter_row, middle_col:three_quarters_col, 13] = predictions[row, col, :quarter_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:quarter_row, three_quarters_col:end_col, 12] = predictions[row, col, :quarter_patch_side, three_quarters_patch_side:]
                    unified_predictions[quarter_row:middle_row, start_col:quarter_col, 5] = predictions[row, col, quarter_patch_side:half_patch_side, :quarter_patch_side]
                    unified_predictions[quarter_row:middle_row, quarter_col:middle_col, 7] = predictions[row, col, quarter_patch_side:half_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[quarter_row:middle_row, middle_col:three_quarters_col, 9] = predictions[row, col, quarter_patch_side:half_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[quarter_row:middle_row, three_quarters_col:end_col, 8] = predictions[row, col, quarter_patch_side:half_patch_side, three_quarters_patch_side:]
                    unified_predictions[middle_row:three_quarters_row, start_col:quarter_col, 3] = predictions[row, col, half_patch_side:three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[middle_row:three_quarters_row, quarter_col:middle_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[middle_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, half_patch_side:three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[middle_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, :three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]
                    
                elif row > 2 and col == 2:

                    unified_predictions[start_row:quarter_row, start_col:quarter_col, 11] = predictions[row, col, :quarter_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:quarter_row, quarter_col:middle_col, 14] = predictions[row, col, :quarter_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:quarter_row, middle_col:three_quarters_col, 13] = predictions[row, col, :quarter_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:quarter_row, three_quarters_col:end_col, 12] = predictions[row, col, :quarter_patch_side, three_quarters_patch_side:]
                    unified_predictions[quarter_row:middle_row, start_col:quarter_col, 8] = predictions[row, col, quarter_patch_side:half_patch_side, :quarter_patch_side]
                    unified_predictions[quarter_row:middle_row, quarter_col:middle_col, 10] = predictions[row, col, quarter_patch_side:half_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[quarter_row:middle_row, middle_col:three_quarters_col, 9] = predictions[row, col, quarter_patch_side:half_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[quarter_row:middle_row, three_quarters_col:end_col, 8] = predictions[row, col, quarter_patch_side:half_patch_side, three_quarters_patch_side:]
                    unified_predictions[middle_row:three_quarters_row, start_col:quarter_col, 5] = predictions[row, col, half_patch_side:three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[middle_row:three_quarters_row, quarter_col:middle_col, 6] = predictions[row, col, half_patch_side:three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[middle_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, half_patch_side:three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[middle_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:middle_col, 2] = predictions[row, col, three_quarters_patch_side:, :half_patch_side]
                    unified_predictions[three_quarters_row:end_row, middle_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, half_patch_side:three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]
             
                else:
                    unified_predictions[start_row:quarter_row, start_col:quarter_col, 15] = predictions[row, col, :quarter_patch_side, :quarter_patch_side]
                    unified_predictions[start_row:quarter_row, quarter_col:middle_col, 14] = predictions[row, col, :quarter_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[start_row:quarter_row, middle_col:three_quarters_col, 13] = predictions[row, col, :quarter_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[start_row:quarter_row, three_quarters_col:end_col, 12] = predictions[row, col, :quarter_patch_side, three_quarters_patch_side:]
                    unified_predictions[quarter_row:middle_row, start_col:quarter_col, 11] = predictions[row, col, quarter_patch_side:half_patch_side, :quarter_patch_side]
                    unified_predictions[quarter_row:middle_row, quarter_col:middle_col, 10] = predictions[row, col, quarter_patch_side:half_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[quarter_row:middle_row, middle_col:three_quarters_col, 9] = predictions[row, col, quarter_patch_side:half_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[quarter_row:middle_row, three_quarters_col:end_col, 8] = predictions[row, col, quarter_patch_side:half_patch_side, three_quarters_patch_side:]
                    unified_predictions[middle_row:three_quarters_row, start_col:quarter_col, 7] = predictions[row, col, half_patch_side:three_quarters_patch_side, :quarter_patch_side]
                    unified_predictions[middle_row:three_quarters_row, quarter_col:middle_col, 6] = predictions[row, col, half_patch_side:three_quarters_patch_side, quarter_patch_side:half_patch_side]
                    unified_predictions[middle_row:three_quarters_row, middle_col:three_quarters_col, 5] = predictions[row, col, half_patch_side:three_quarters_patch_side, half_patch_side:three_quarters_patch_side]
                    unified_predictions[middle_row:three_quarters_row, three_quarters_col:end_col, 4] = predictions[row, col, half_patch_side:three_quarters_patch_side, three_quarters_patch_side:]
                    unified_predictions[three_quarters_row:end_row, start_col:quarter_col, 3] = predictions[row, col, three_quarters_patch_side:, :quarter_patch_side]
                    unified_predictions[three_quarters_row:end_row, quarter_col:middle_col, 2] = predictions[row, col, three_quarters_patch_side:, quarter_patch_side:half_patch_side]
                    unified_predictions[three_quarters_row:end_row, middle_col:three_quarters_col, 1] = predictions[row, col, three_quarters_patch_side:, half_patch_side:three_quarters_patch_side]
                    unified_predictions[three_quarters_row:end_row, three_quarters_col:end_col, 0] = predictions[row, col, three_quarters_patch_side:, three_quarters_patch_side:]
                    
    unified_predictions = unified_predictions[int(patch_side - patch_step):int(start_row + patch_step), int(patch_side - patch_step):int(start_col + patch_step)]

    return unified_predictions
