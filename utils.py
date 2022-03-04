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
