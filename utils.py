import numpy as np
import matplotlib.pyplot as plt

def center_crop(img: np.array, crop_shape: tuple):
   
    crop_rows, crop_cols = crop_shape[:2]

    if len(img.shape) == 2:
        img_rows, img_cols = img.shape

        start_row = int((img_rows - crop_rows) / 2)    
        start_col = int((img_cols - crop_cols) / 2)
        end_row = start_row + crop_rows
        end_col = start_col + crop_cols

        cropped_img = img[start_row:end_row, start_col:end_col]

    elif len(img.shape) == 3:
        img_rows, img_cols, _ = img.shape

        start_row = int((img_rows - crop_rows) / 2)
        start_col = int((img_cols - crop_cols) / 2)
        end_row = start_row + crop_rows
        end_col = start_col + crop_cols

        cropped_img = img[start_row:end_row, start_col:end_col, :]

    else:
        _, img_rows, img_cols, _ = img.shape
    
        start_row = int((img_rows - crop_rows) / 2)    
        start_col = int((img_cols - crop_cols) / 2)
        end_row = start_row + crop_rows
        end_col = start_col + crop_cols

        cropped_img = img[:, start_row:end_row, start_col:end_col, :]

    return cropped_img 


def rebuild_from_patches(predictions, unified_shape, patch_step, patch_side):

    overlap_ratio = patch_side / patch_step
    assert overlap_ratio in (2, 4), "The patch step can only be 1/2 or 1/4 of the patch side" 

    patches_num_vertically = predictions.shape[0]
    patches_num_horizontally = predictions.shape[1]
    
    patch_side = int(patch_side)
    half_patch_side = int(patch_side / 2)

    unified_predictions = np.empty(unified_shape)
    unified_predictions[:] = np.NaN

    if overlap_ratio == 2:
        for row in range(patches_num_vertically):
            for col in range(patches_num_horizontally):
                
                zero_row = patch_step * row
                start_row = zero_row
                middle_row = zero_row + half_patch_side
                end_row = zero_row + patch_side

                zero_col = patch_step * col
                start_col = zero_col
                middle_col = zero_col + half_patch_side
                end_col = zero_col + patch_side


                if row == 0 and col == 0:
                    unified_predictions[:patch_side, :patch_side, 0] = predictions[row, col]
                
                elif row == 0:
                    unified_predictions[zero_row:end_row, start_col:middle_col, 1] = predictions[row, col, :, :half_patch_side]
                    unified_predictions[zero_row:end_row, middle_col:end_col, 0] = predictions[row, col, :, half_patch_side:]

                elif col == 0:
                    unified_predictions[zero_row:middle_row, start_col:middle_col, 1] = predictions[row, col, :half_patch_side, :half_patch_side]
                    unified_predictions[zero_row:middle_row, middle_col:end_col, 2] = predictions[row, col, :half_patch_side, half_patch_side:]
                    unified_predictions[middle_row:end_row, start_col:end_col, 0] = predictions[row, col, half_patch_side:, :]

                else:
                    unified_predictions[zero_row:middle_row, start_col:middle_col, 3] = predictions[row, col, :half_patch_side, :half_patch_side]
                    unified_predictions[zero_row:middle_row, middle_col:end_col, 2] = predictions[row, col, :half_patch_side, half_patch_side:]
                    unified_predictions[middle_row:end_row, start_col:middle_col, 1] = predictions[row, col, half_patch_side:, :half_patch_side]
                    unified_predictions[middle_row:end_row, middle_col:end_col, 0] = predictions[row, col, half_patch_side:, half_patch_side:]
    else:
        pass
    """
    elif overlap_ratio == 4:
        quarter_patch_side = int(patch_side / 4)
        three_quarters_patch_side = int(quarter_patch_side * 3)

        for row in range(patches_num_vertically):
            for col in range(patches_num_horizontally):

                quarter_row = zero_row + quarter_patch_side
                three_quarters_row = zero_row + three_quarters_patch_side

                quarter_col = zero_col + quarter_patch_side
                three_quarters_col = zero_col + three_quarters_patch_side

    """
    return unified_predictions
