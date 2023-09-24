import os
import sys

from argparse import ArgumentParser
import torch
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class IAMHandwrittenDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join('/dataset/IAM/', "ruled_data")
        self.masks_directory = os.path.join('/dataset/IAM/', "masks")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        cwd = os.getcwd()
        filename = self.filenames[idx]
        #image_path = os.path.join(self.images_directory, filename + ".png")
        #mask_path = os.path.join(self.masks_directory, filename + ".png")
        image_path = cwd+os.path.join(self.images_directory, filename)
        mask_path = cwd+os.path.join(self.masks_directory, filename)

        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path).convert('L'))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        #print(type(mask))
        #print(np.unique(mask))
        #contain only two uniques values:
        # 0.0 if a pixel is a background and 1.0 if a pixel is a line .
        mask[mask == 0.0] = 0.0
        mask[(mask == 255.0)] = 1.0
        #print(np.unique(mask))
        #exit()
        return mask

    def _read_split(self):
        cwd = os.getcwd()
        img_dir = cwd+self.images_directory
        filenames = []
        for dir_path, dir_names, file_names in os.walk(img_dir):
            if self.mode == "train":  # 90% for train
                filenames = [x for i, x in enumerate(file_names) if i % 10 != 0]
            elif self.mode == "valid":  # 10% for validation
                filenames = [x for i, x in enumerate(file_names) if i % 10 == 0]

        return filenames




class SimpleIAMHandwrittenDataset(IAMHandwrittenDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample
