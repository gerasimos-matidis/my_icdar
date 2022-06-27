from tkinter.filedialog import askdirectory
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

d = askdirectory(initialdir='./models')
print(d)
validation_pairs = np.load(os.path.join(d, 'validation_pairs.npy'), allow_pickle=True).tolist()
predictions_info = np.load(os.path.join(d, 'predictions.npy'), allow_pickle=True).tolist()
x = validation_pairs['x']
y = validation_pairs['y']
pred_ims = np.array([np.where(predictions_info[i]['model output'] > 0.5, 1, 0).astype(int) for i in range(len(predictions_info))])
loss = np.array([predictions_info[i]["loss"] for i in range(len(predictions_info))]).squeeze()
mean_iou = np.array([predictions_info[i]["mean_iou"] for i in range(len(predictions_info))]).squeeze()
predictions_num = len(predictions_info)
PLOT_ROWS_NUM = 2
plot_cols_num = int(np.ceil((predictions_num + 2) / PLOT_ROWS_NUM))
colormap = 'gray'

fig, ax = plt.subplots(PLOT_ROWS_NUM, plot_cols_num)
im = ax[0, 0].imshow(x[0])
ax[0, 0].title.set_text('Input image')
target = ax[1, 0].imshow(y[0], cmap=colormap)
ax[1, 0].title.set_text('Ground truth')

row, col = 1, 0
for i in range(predictions_num):
    if row + 1 < PLOT_ROWS_NUM:
        row += 1
    else:
        col += 1
        row = 0
    exec(f'p{i} = ax[row, col].imshow(pred_ims[i][0], cmap=colormap)')
    ax[row, col].title.set_text(f'{predictions_info[i]["mode"]}\n{predictions_info[i]["initial images"]} images, {predictions_info[i]["epochs"]} epochs')
    
axdepth = plt.axes([0.1, 0.01, 0.8, 0.03])
sliderdepth = Slider(axdepth, '', 0, x.shape[0] - 1, valinit=0, valstep=1)

def slider_update(val):
    im.set_data(x[val])
    target.set_data(y[val])

    row, col = 1, 0
    for i in range(predictions_num):
        if row + 1 < PLOT_ROWS_NUM:
                row += 1
        else:
            col += 1
            row = 0
        
        eval(f'p{i}.set_data(pred_ims[i][val])')
        ax[row, col].set(xlabel=f'loss: {"{:0.3f}".format(loss[i][val])}\nmean IoU: {"{:0.2f}".format(mean_iou[i][val])}')
        #ax[row, col].set(xlabel=f'loss: {"{:0.4f}".format(loss[i][val])}')
                
sliderdepth.on_changed(slider_update)
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()
