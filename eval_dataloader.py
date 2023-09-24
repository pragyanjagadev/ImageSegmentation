import glob

from torch.utils import data
import os
from PIL import Image

from torch.utils.data import Dataset


class EvalDataset(Dataset):
    def __init__(self, img_dir, pred_dir, scale=1):
        self.img_dir = img_dir
        self.pred_dir = pred_dir
        self.scale = scale

        try:
            self.ids = [s for s in os.listdir(self.img_dir)]
        except FileNotFoundError:
            self.ids = []

        self.image_path = list(
            map(lambda x: os.path.join(self.img_dir, x), self.ids))

        self.label_path = list(
            map(lambda x: os.path.join(self.pred_dir, x), self.ids))

    def __getitem__(self, item):
        #print(self.image_path[0])
        #print(self.label_path[0])
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
