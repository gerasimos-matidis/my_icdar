import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

predicions_info = np.load('predictions.npy', allow_pickle=True).tolist()
validation_pairs = np.load('validation_pairs.npy', allow_pickle=True).tolist()
x = validation_pairs['x']
y = validation_pairs['y']

pred_ims = [predicions_info[i]['model_output'] for i in range(len(predicions_info))]

print(predicions_info[1]['method'])
print(np.max(pred_ims[1]))
colormap = 'gray'
fig, ax = plt.subplots(2, 3)


im = ax[0, 0].imshow(x[0])
target = ax[0, 1].imshow(y[0], cmap=colormap)
p1 = ax[1, 0].imshow(pred_ims[0][0], cmap=colormap)
p2 = ax[1, 1].imshow(pred_ims[1][0], cmap=colormap)

axdepth = plt.axes([0.1, 0.01, 0.8, 0.03])
sliderdepth = Slider(axdepth, '', 0, x.shape[0] - 1, valinit=0, valstep=1)

def slider_update(val):
    im.set_data(x[val])
    target.set_data(y[val])
    p1.set_data(pred_ims[0][val])
    p2.set_data(pred_ims[1][val])
    
sliderdepth.on_changed(slider_update)

plt.show()
