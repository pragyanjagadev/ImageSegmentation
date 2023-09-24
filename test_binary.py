import io
import os
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch
from unet_binary import UnetBinary
from dataset_binary import DirDataset
from functions import iou_score_numpy

def predict(net, img, device='cpu', threshold=0.5):
    ds = DirDataset('', '')

    _img = torch.from_numpy(ds.preprocess(img))

    _img = _img.unsqueeze(0)
    _img = _img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        o = net(_img)

        if net.n_classes > 1:
            pass
        else:
            probs = torch.sigmoid(o)
        #print(probs)
        #print(probs.shape)
        #sys.exit()
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img.size[1]),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        #print(mask)
    return mask > threshold


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def main(hparams):
    ds = DirDataset('', '')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #checkpoint = './lightning_logs_binary/version_0/epoch=0_val_loss=0.15.ckpt'
    net = UnetBinary().load_from_checkpoint(hparams.checkpoint)
    net.freeze()
    net.to(device)

    for fn in tqdm(os.listdir(hparams.img_dir)):
        fp = os.path.join(hparams.img_dir, fn)
        img = Image.open(fp).convert("L")

        pred_mask = predict(net, img, device=device)
        pred_mask_img = mask_to_image(pred_mask)

        pred_mask_img.save(os.path.join(hparams.out_dir, fn))
        #sys.exit()

def evaluate():

    ds = DirDataset('', '')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    avg_iou_score, img_num = 0.0, 0.0

    for fn in tqdm(os.listdir(hparams.out_dir)):
        fp = os.path.join(hparams.out_dir, fn)
        mask_path = os.path.join(hparams.mask_dir, fn)
        mask = Image.open(mask_path).convert("L")
        mask_arr = ds.preprocess(mask)
        pred_mask_img = Image.open(fp).convert("L")
        pred_mask_img_arr = ds.preprocess(pred_mask_img)
        img_num += 1.0
        #get IoU score
        iou_score = iou_score_numpy(mask_arr, pred_mask_img_arr)
        avg_iou_score += iou_score
        print(iou_score)

    avg_iou_score_final = avg_iou_score / img_num
    print(avg_iou_score_final)
    #sys.exit()

if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--checkpoint', default='./lightning_logs_binary/version_0/epoch=0_val_loss=0.15.ckpt')
    parent_parser.add_argument('--img_dir', default="./dataset/IAM/input")
    parent_parser.add_argument('--out_dir', default="./dataset/IAM/mask_binary_pred")
    parent_parser.add_argument('--mask_dir', default="./dataset/IAM/masks")
    parser = UnetBinary.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()
    main(hparams)
    evaluate()
