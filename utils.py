import numpy as np
import matplotlib.pyplot as plt

def center_crop(img: np.array, crop_shape: tuple):
   
    crop_h, crop_w = crop_shape[:2]

    if len(img.shape) == 2:
        img_h, img_w = img.shape

        start_h = int((img_h - crop_h) / 2)    
        start_w = int((img_w - crop_w) / 2)
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        cropped_img = img[start_h:end_h, start_w:end_w]

    elif len(img.shape) == 3:
        img_h, img_w, _ = img.shape

        start_h = int((img_h - crop_h) / 2)    
        start_w = int((img_w - crop_w) / 2)
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        cropped_img = img[start_h:end_h, start_w:end_w, :]

    else:
        _, img_h, img_w, _ = img.shape
    
        start_h = int((img_h - crop_h) / 2)    
        start_w = int((img_w - crop_w) / 2)
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        cropped_img = img[:, start_h:end_h, start_w:end_w, :]

    return cropped_img 



def reconnect_patches(predictions, unified_shape, patch_step, input_size):

    # NOTE: The folowing code works only when the step to create patches (see above) is set to half of the patch size.
    # I need to rewrite is (as a function) so that it will work in general
    patches_num_v = predictions.shape[0]
    patches_num_h = predictions.shape[1]
    patch_side = int(input_size)
    half_patch_side = int(patch_side/2)

    unified_predictions = np.empty(unified_shape)
    unified_predictions[:] = np.NaN

    for v in range(patches_num_v):
        for h in range(patches_num_h):

            v_zero = patch_step * v
            start_v = v_zero
            middle_v = v_zero + half_patch_side
            end_v = v_zero + patch_side
            
            h_zero = patch_step * h
            start_h = h_zero
            middle_h = h_zero + half_patch_side
            end_h = h_zero + patch_side
            
            if v == 0 and h == 0:
                unified_predictions[0:patch_side, 0:patch_side, 0] = predictions[v][h]
            
            elif v == 0:
                unified_predictions[start_v:end_v, start_h:middle_h, 1] = predictions[v][h][:, 0:half_patch_side]
                unified_predictions[start_v:end_v, middle_h:end_h, 0] = predictions[v][h][:, half_patch_side:]

            elif h == 0:
                unified_predictions[start_v:middle_v, start_h:middle_h, 1] = predictions[v][h][0:half_patch_side, 0:half_patch_side]
                unified_predictions[start_v:middle_v, middle_h:end_h, 2] = predictions[v][h][0:half_patch_side, half_patch_side:]
                unified_predictions[middle_v:end_v, start_h:end_h, 0] = predictions[v][h][half_patch_side:, :]

            else:
                unified_predictions[start_v:middle_v, start_h:middle_h, 3] = predictions[v][h][0:half_patch_side, 0:half_patch_side]
                unified_predictions[start_v:middle_v, middle_h:end_h, 2] = predictions[v][h][0:half_patch_side, half_patch_side:]
                unified_predictions[middle_v:end_v, start_h:middle_h, 1] = predictions[v][h][half_patch_side:, 0:half_patch_side]
                unified_predictions[middle_v:end_v, middle_h:end_h, 0] = predictions[v][h][half_patch_side:, half_patch_side:]

    return unified_predictions


    # just to check
        