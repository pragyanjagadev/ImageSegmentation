import sys
import cv2
import numpy as np
import random
import os
from PIL import Image
import os
import warnings
from functions import lineHeights

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class GenerateRules:
    def __init__(self, iam_data_path, cropped_data_path, iam_ruled_data_path, iam_segmented_data,
                 iam_removed_rule_data):

        self.iam_data_path = iam_data_path
        self.cropped_data_path = cropped_data_path
        self.iam_ruled_data_path = iam_ruled_data_path
        self.iam_segmented_data = iam_segmented_data
        self.iam_removed_rule_data = iam_removed_rule_data

    def projectionAnalysis(self, im):
        # compute the ink density histogram (sum each rows)
        hist = cv2.reduce(im, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
        hist = hist.ravel()
        # find peaks withing the ink density histogram
        max_hist = max(hist)
        mean_hist = np.mean(hist)
        thres_hist = mean_hist / max_hist
        peaks = peakutils.indexes(hist, thres=thres_hist, min_dist=50)
        # find peaks that are too high
        mean_peaks = np.mean(hist[peaks])
        std_peaks = np.std(hist[peaks])
        thres_peaks_high = mean_peaks + 1.5 * std_peaks
        thres_peaks_low = mean_peaks - 3 * std_peaks
        peaks = peaks[np.logical_and(hist[peaks] < thres_peaks_high, hist[peaks] > thres_peaks_low)]

        return peaks

    def centerCrop(self, img, file):
        """
        Cropping the image from center
        :Parameters:
          img : an image name
          path : respective path of image
        """
        cropped_file = self.cropped_data_path + '/' + file
        width = img.shape[1]
        height = img.shape[0]

        new_width = min(width, height)
        new_height = min(width, height)

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 1.5))
        bottom = height - int(np.floor((height - new_height)))

        if len(img.shape) == 2:
            center_cropped_img = img[top:bottom, left:right]
        else:
            center_cropped_img = img[top:bottom, left:right, ...]

        cv2.imwrite(cropped_file, center_cropped_img)

    def drawRules(self):
        """
           Draw line/rule after cropping the image dynamically and save it
        """
        for subdir, dirs, files in os.walk(self.iam_data_path):

            for file in files:
                if file != '.DS_Store':
                    # print(os.path.join(subdir, file))
                    input_file = self.iam_data_path + '/' + file
                    output_file = self.iam_ruled_data_path + '/' + file
                    img = cv2.imread(input_file)
                    # croping image
                    self.centerCrop(img, file)
                    cropped_file = self.cropped_data_path + '/' + file
                    cropped_img = cv2.imread(cropped_file)  #
                    origimg = np.array(Image.open(cropped_file))

                    lineHeights(origimg, output_file, False, True)

                    # remove lines
                    removed_rule_file = self.iam_removed_rule_data + '/' + file
                    # print(output_file)
                    # lineRemove(output_file, removed_rule_file)

    def removeRules(self):
        """
           Draw line/rule after cropping the image dynamically and save it
        """
        for subdir, dirs, files in os.walk(self.iam_data_path):

            print(self.iam_data_path)
            print(files)
            for file in files:
                print("------------")
                if file != '.DS_Store':
                    print("==========")
                    # print(os.path.join(subdir, file))
                    input_file = self.iam_data_path + '/' + file
                    print(input_file)
                    output_file = self.iam_ruled_data_path + '/' + file
                    img = cv2.imread(input_file)
                    # croping image
                    self.centerCrop(img, file)
                    cropped_file = self.cropped_data_path + '/' + file
                    cropped_img = cv2.imread(cropped_file)  #
                    origimg = np.array(Image.open(cropped_file))

                    lineHeights(origimg, output_file, False, True)

                    # remove lines
                    removed_rule_file = self.iam_removed_rule_data + '/' + file
                    # print(output_file)
                    # lineRemove(output_file, removed_rule_file)


def main():
    cwd = os.getcwd()
    iam_data = cwd + '/dataset/IAM/formsI-Z'
    cropped_data = cwd + '/dataset/IAM/cropped_data'
    iam_ruled_data = cwd + '/dataset/IAM/ruled_data'
    iam_augmented_data = cwd + '/dataset/IAM/augmented_data'
    iam_segmented_data = cwd + '/dataset/IAM/segmented'
    iam_removed_rule_data = cwd + '/dataset/IAM/removedrule'

    # obj_rule = GenerateRules(iam_data, cropped_data, iam_ruled_data, iam_segmented_data, iam_removed_rule_data)
    # obj_rule.drawRules()

    obj_rule = GenerateRules(iam_augmented_data, iam_removed_rule_data)
    obj_rule.removeRules()


if __name__ == '__main__':
    main()
