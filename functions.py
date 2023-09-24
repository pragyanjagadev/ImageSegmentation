import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

def lineHeights(origimg, filename, show=False, write=False, return_img=False):

    if origimg.ndim == 3:
        origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2GRAY)

    img = origimg.astype(float)
    if np.max(img) > 1:
        img /= 255.0

    mean_height = 0.0
    std_height = 1.0
    num_lines = 0
    col_range = None

    # col_range = (0,int(img.shape[1] / 4))
    # col_range = (int(img.shape[1]/8.0),int(img.shape[1] / 3.0))
    # profile = np.sum(img[:,col_range[0]:col_range[1]], axis=1)
    col_div = 3
    for i in range(col_div):
        c_col_range = (int((float(i) / col_div) * img.shape[1] * 0.45),
                       int((float(i + 1) / col_div) * img.shape[1] * 0.45))
        profile = np.sum(img[:, c_col_range[0]:c_col_range[1]], axis=1)
        # find best sigma value
        for s in range(21, 302, 10):
            c_g_profile = cv2.GaussianBlur(profile, (s, s), 0)
            # central difference
            minima = np.where((c_g_profile[1:-1] < c_g_profile[:-2]) \
                              & (c_g_profile[1:-1] < c_g_profile[2:]))[0]
            # get local minima (except first and last line):
            heights = minima[1:] - minima[:-1]
            heights = heights[1:-1]
            if len(heights) == 0:
                break
            c_mean_height = np.mean(heights)
            c_std_height = np.std(heights)

            if len(minima) > 8 and len(minima) < 50 \
                    and c_mean_height / c_std_height > mean_height / std_height:
                num_lines = len(minima) + 1
                #and c_std_height < std_height:
                mean_height = c_mean_height
                std_height = c_std_height
                col_range = c_col_range
                g_profile = c_g_profile

        # if no proper lines found so far it's probably only
        # an excerpt of a charter
        if not col_range:
            for s in range(7, 308, 10):
                c_g_profile = cv2.GaussianBlur(profile, (s, s), 0)
                # central difference
                minima = np.where((c_g_profile[1:-1] < c_g_profile[:-2]) \
                                  & (c_g_profile[1:-1] < c_g_profile[2:]))[0]
                # get local minima (except first and last line):
                heights = minima[1:] - minima[:-1]

                if len(heights) == 0:
                    break
                c_mean_height = np.median(heights)
                c_std_height = np.std(heights)

                if c_mean_height / c_std_height > mean_height / std_height:
                    num_lines = len(minima) + 1
                    #           and c_std_height < std_height:
                    mean_height = c_mean_height
                    std_height = c_std_height
                    col_range = c_col_range
                    g_profile = c_g_profile

    if (show or write or return_img) and col_range != None:
        line_img = origimg.copy()
        # line_img = np.bitwise_not(line_img)
        line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
        # line_img[:, col_range[0]:col_range[1]] /= 2
        # draw line boundings
        maxima = np.where((g_profile[1:-1] > g_profile[:-2]) \
                          & (g_profile[1:-1] > g_profile[2:]))[0]
        minima = np.where((g_profile[1:-1] < g_profile[:-2]) \
                          & (g_profile[1:-1] < g_profile[2:]))[0]

        for maxi in maxima:
            color = list(np.random.random(size=3) * 256)
            cv2.line(line_img, (0, maxi), (img.shape[1], maxi), color, 3)

        # draw line masses
        minima = np.where((g_profile[1:-1] < g_profile[:-2]) \
                          & (g_profile[1:-1] < g_profile[2:]))[0]
        # for mini in minima:
        #    cv2.line(line_img, (0, mini), (img.shape[1], mini), (100, 0, 255), 3)

        show_img = cv2.resize(line_img,
                              (int((800.0 / line_img.shape[0]) * \
                                   line_img.shape[1]), 800))
        #                              show_img, 0.2, 0.2)
        if show:
            cv2.imshow('lines', show_img)
            cv2.waitKey()
        if write:
            cv2.imwrite(filename, show_img)

    if return_img:
        return num_lines, mean_height, std_height, line_img

    return num_lines, mean_height, std_height, None


def iou_score_numpy(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    #print(iou_score)
    return iou_score

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
