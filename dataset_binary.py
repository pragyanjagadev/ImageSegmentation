import os
import glob
import sys

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class DirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, scale=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.scale = scale

        try:
            self.ids = [s.split('.')[0] for s in os.listdir(self.img_dir)]
        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, is_mask=False):
        w, h = img.size
        _h = 800
        _w = 800
        assert _w > 0
        assert _h > 0

        if not is_mask:
            _img = img.resize((_w, _h))
            _img = np.array(_img)
            if len(_img.shape) == 2:  # Gray/mask images
                _img = np.expand_dims(_img, axis=-1)

            # HWC to CHW
            _img = _img.transpose((2, 0, 1))
            if _img.max() > 1:
                _img = _img / 255.
        else:
            _img = img.resize((_w, _h), Image.NEAREST)

        return _img

    def __getitem__(self, i):
        # ...
        idx = self.ids[i]
        mask_idx = idx.split("_aug")[0] if len(idx.split("_aug")) > 1 else idx
        img_files = glob.glob(os.path.join(self.img_dir, idx + '.*'))
        mask_files = glob.glob(os.path.join(self.mask_dir, mask_idx + '.*'))

        assert len(img_files) == 1, f'{idx}: {img_files}'
        assert len(mask_files) == 1, f'{idx}: {mask_files}'

        img = Image.open(img_files[0]).convert("L")
        mask = Image.open(mask_files[0]).convert("L")  # Ensure it's grayscale
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        #print("============")
        #print(type(img), type(mask))
        #print(img.size)
        #print(mask.size)
        img_tensor = transform(img)
        mask_tensor = transform(mask)

        #print("============")
        #print(type(img_tensor), type(mask_tensor))
        #print(img_tensor.shape)
        #print(mask_tensor.shape)

        # assert img.size == mask.size, f'{img.shape} # {mask.shape}'

        # print(mask_tensor.shape)
        # sys.exit()
        # print(type(img))
        # print(type(mask))
        img = self.preprocess(img)
        mask = self.preprocess(mask)


        return torch.from_numpy(img).float(), \
            torch.from_numpy(mask).float()