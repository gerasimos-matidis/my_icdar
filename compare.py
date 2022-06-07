import os
import numpy as np
import matplotlib.pyplot as plt

models_info = []
for dirpath, dirname, filenames in os.walk('trained_models/full_001'):
    for f in filenames:
        if f == 'history.npy':
            history = np.load(os.path.join(dirpath, f), allow_pickle=True).tolist()
            split_dirpath = dirpath.split('_')[5:8]

            d = {
                'method': split_dirpath[0] + ' ' + split_dirpath[1],
                'initial images': split_dirpath[2].replace('initialimagesnumber', ''),
                'loss': history['loss']
            }

            models_info.append(d)   

for i in range(len(models_info)):
    xs = list(range(len(models_info[i]['loss'])))
    plt.plot(xs, models_info[i]['loss'], label=f'{models_info[i]["method"]}, {models_info[i]["initial images"]} initial images')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
