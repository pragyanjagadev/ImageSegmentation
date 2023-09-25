#!pip install -r requirements.txt --quiet
#!pip install wget --quiet

import os
import base64, os
from IPython.display import HTML, Image
from base64 import b64decode
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd()
cwd
images_path = cwd + '/dataset/IAM/ruled_data'
masks_path = cwd + '/dataset/IAM/masks_binary'

for subdir, dirs, images in os.walk(images_path):
    for img in images:
        fname = images_path + '/' + img
        mask_path = masks_path + '/' + img
        image64 = base64.b64encode(open(fname, 'rb').read())
        image64 = image64.decode('utf-8')

        print(f'Will use {fname} for inpainting')
        img = np.array(plt.imread(f'{fname}')[:,:,:3])
        print(f'Will use {mask_path} for inpainting')
        mask = np.array(plt.imread(f'{mask_path}')[:,:,:3])
        print('Run inpainting')
        if '.jpeg' in fname:
            !PYTHONPATH=. TORCH_HOME=$(pwd) python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=output dataset.img_suffix=.jpeg > /dev/null
        elif '.jpg' in fname:
            !PYTHONPATH=. TORCH_HOME=$(pwd) python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=output  dataset.img_suffix=.jpg > /dev/null
        elif '.png' in fname:
            !PYTHONPATH=. TORCH_HOME=$(pwd) python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=output  dataset.img_suffix=.png > /dev/null
        else:
            print(f'Error: unknown suffix .{fname.split(".")[-1]} use [.png, .jpeg, .jpg]')


#%%
