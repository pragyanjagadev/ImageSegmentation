import sys
import cv2
import numpy as np
import random
import os
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

class GetMasks:
    def __init__(self, iam_augmented_data, iam_segmented_data):

        self.iam_augmented_data = iam_augmented_data
        self.iam_segmented_data = iam_segmented_data

    def getMasks(self):
        for subdir, dirs, files in os.walk(self.iam_augmented_data):
            for file in files:
                if file != '.DS_Store':
                    input_file = self.iam_augmented_data + '/' + file
                    output_file = self.iam_segmented_data + '/' + file
                    img = cv2.imread(input_file)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_black = np.array([0, 42, 0])
                    upper_black = np.array([179, 255, 255])
                    mask = cv2.inRange(hsv, lower_black, upper_black)
                    cv2.imwrite(output_file, mask)


def main():
    cwd = os.getcwd()
    #iam_augmented_data = cwd + '/dataset/IAM/augmented_data'
    #iam_segmented_data = cwd + '/dataset/IAM/segmented'
    iam_augmented_data = cwd + '/dataset/IAM/cropped_data'
    iam_segmented_data = cwd + '/dataset/IAM/masks_29'

    obj_mask = GetMasks(iam_augmented_data, iam_segmented_data)
    obj_mask.getMasks()


if __name__ == '__main__':
    main()
