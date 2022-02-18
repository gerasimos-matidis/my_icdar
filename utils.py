import numpy as np
import matplotlib.pyplot as plt

def center_crop(skipped_fmaps: np.array, crop_shape: tuple):

    print('here', np.shape(skipped_fmaps), 'there', crop_shape)
    
    crop_h, crop_w = crop_shape[:2]
    _, fmaps_h, fmaps_w, _ = np.shape(skipped_fmaps)

    start_h = int((fmaps_h - crop_h) / 2)    
    start_w = int((fmaps_w - crop_w) / 2)

    end_h = start_h + crop_h
    end_w = start_w + crop_w

    aa = skipped_fmaps[:, start_h:end_h, start_w:end_w, :]

    print('gia pes', aa.shape)


    return aa 
