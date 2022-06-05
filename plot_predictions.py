import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

predicions_info = np.load('predictions.npy', allow_pickle=True).tolist()
validation_pairs = np.load('validation_pairs.npy', allow_pickle=True).tolist()
x = validation_pairs['x']
y = validation_pairs['y']

pred_ims = [np.where(predicions_info[i]['model output'] > 0.5, 1, 0).astype(int) for i in range(len(predicions_info))]
colormap = 'gray'

fig, ax = plt.subplots(2, 3)
im = ax[0, 0].imshow(x[0])
target = ax[0, 1].imshow(y[0], cmap=colormap)
# TODO: create p_i with automatically using a for loop
p1 = ax[0, 2].imshow(pred_ims[0][0], cmap=colormap)
p2 = ax[1, 0].imshow(pred_ims[1][0], cmap=colormap)
p3 = ax[1, 1].imshow(pred_ims[2][0], cmap=colormap)
p4 = ax[1, 2].imshow(pred_ims[3][0], cmap=colormap)

axdepth = plt.axes([0.1, 0.01, 0.8, 0.03])
sliderdepth = Slider(axdepth, '', 0, x.shape[0] - 1, valinit=0, valstep=1)

def slider_update(val):
    im.set_data(x[val])
    target.set_data(y[val])
    p1.set_data(pred_ims[0][val])
    p2.set_data(pred_ims[1][val])
    p3.set_data(pred_ims[2][val])
    p4.set_data(pred_ims[3][val])
    
sliderdepth.on_changed(slider_update)
plt.show()
