
import io
import os
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

import torch
from unetbinary11 import Unet
from dataset import DirDataset


def predict(net, img, device='cpu', threshold=0.5):
    ds = DirDataset('', '')

    _img = torch.from_numpy(ds.preprocess(img))

    _img = _img.unsqueeze(0)
    _img = _img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        o = net(_img)

        if net.n_classes > 1:
            print("=======")
            probs = torch.argmax(o, dim=1)
        else:
            probs = torch.sigmoid(o)
        #print(probs)
        print(probs.shape)
        #sys.exit()
        #probs = probs.squeeze(0)

        mask = probs.squeeze().cpu().numpy()
        print(mask)
    return mask > threshold


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = hparams.checkpoint
    net = Unet().load_from_checkpoint(checkpoint)
    net.freeze()
    net.to(device)
    img_dir = "./dataset/IAM/input"
    out_dir = "./dataset/IAM/output_multiclass"
    for fn in tqdm(os.listdir(img_dir)):
        fp = os.path.join(img_dir, fn)
        img = Image.open(fp).convert('RGB')
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        img_tensor = transform(img)
        #print(img)
        #print(img_tensor.shape)
        mask = predict(net, img, device=device)
        mask_img = mask_to_image(mask)
        mask_img.save(os.path.join(out_dir, fn))
        sys.exit()

if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    #parent_parser.add_argument('--checkpoint', default='version_9/epoch=0_val_loss=2.42.ckpt')
    parent_parser.add_argument('--checkpoint', default='version_9/epoch=0_val_loss=2.42.ckpt')
    parent_parser.add_argument('--img_dir', default="./dataset/IAM/input")
    parent_parser.add_argument('--out_dir', default="./dataset/IAM/test_output")

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()
    main(hparams)
