import os
import numpy as np
from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
from tensorflow import keras
from patchify import patchify, unpatchify
from utils import center_crop

x = askopenfilename(initialdir='.', title='Select input image')
x = plt.imread(x)
x = center_crop(x, (5120, 7680))
y = askopenfilename(title='Select output image')
y = plt.imread(y)
y = center_crop(y, (5120, 7680))


x_patches = patchify(x, (512, 512, 3), step=512)
input_patches = np.reshape(x_patches, ((-1, ) + x_patches.shape[-3:]))
y_patches = patchify(y, (512, 512), step=512)
output_patches = np.reshape(y_patches, ((-1, ) + y_patches.shape[-2:]))

models_path = askdirectory(initialdir='.', title='Select the directory with the models' )
models_list = [name for name in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, name))]

if 'logs' in models_list:
    models_list.remove('logs')
    
if 'predictions' in models_list:
    models_list.remove('predictions')

bce = keras.losses.BinaryCrossentropy()
predictions_info = []
for modelname in models_list:
    print(f'Loading ==> {modelname}')
    model = keras.models.load_model(os.path.join(models_path, modelname))
    splitted_modelname = modelname.split('_')[3:]
    prediction = np.squeeze(model.predict(input_patches))
    reshaped_prediction = np.reshape(prediction, y_patches.shape)
    full_prediction = unpatchify(reshaped_prediction, y.shape)
    binarized_prediction = np.where(full_prediction > 0.5, 1, 0)
    m = keras.metrics.MeanIoU(num_classes=2)
    m.update_state(y, binarized_prediction)
    
    p_info = {
        'image': full_prediction,
        'loss': bce(y, full_prediction).numpy(),
        'mean_iou': m.result().numpy(),
        'mode': splitted_modelname[1] + ' ' + splitted_modelname[2],
        'initial images': splitted_modelname[0].replace('images', ''),
        'epochs': splitted_modelname[-1].replace('eps', '')
    }  
    predictions_info.append(p_info)

predictions_num = len(predictions_info)
PLOT_ROWS_NUM = 2
plot_cols_num = int(np.ceil((predictions_num + 2) / PLOT_ROWS_NUM))
colormap = 'gray'

fig, ax = plt.subplots(PLOT_ROWS_NUM, plot_cols_num)
im = ax[0, 0].imshow(x)
ax[0, 0].title.set_text('Input image')
target = ax[1, 0].imshow(y, cmap=colormap)
ax[1, 0].title.set_text('Ground truth')


row, col = 1, 0
for i in range(predictions_num):
    if row + 1 < PLOT_ROWS_NUM:
        row += 1
    else:
        col += 1
        row = 0
    exec(f'p{i} = ax[row, col].imshow(predictions_info[i]["image"], cmap=colormap)')
    ax[row, col].title.set_text(f'{predictions_info[i]["mode"]},\n{predictions_info[i]["initial images"]} images, {predictions_info[i]["epochs"]} epochs')
    ax[row, col].set(xlabel=f'loss: {"{:0.3f}".format(predictions_info[i]["loss"])}, mean IoU: {"{:0.2f}".format(predictions_info[i]["mean_iou"])}')


plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()