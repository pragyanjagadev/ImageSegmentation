import sys
import random

import numpy as np
import os
import cv2
from PIL import Image
"""
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
noise_typ : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""


class AdditionalArtifacts:

    def __init__(self, iam_ruled_data_path, iam_artifacts_path):

        self.iam_ruled_data_path = iam_ruled_data_path
        self.iam_artifacts_path = iam_artifacts_path


    def additional_artifacts(self):
        for subdir, dirs, images in os.walk(self.iam_ruled_data_path):
            for img in images:
                input_img = self.iam_ruled_data_path + '/' + img
                base_fn = img.replace('.png', '')
                artifacts=['gauss', 'poisson', 's&p']
                image = cv2.imread(input_img)
                for i in range(len(artifacts)):
                    #print(artifacts[i])
                    output_img = base_fn + '_'+artifacts[i]+ '.png'
                    output_img_path = self.iam_artifacts_path + '/' + output_img
                    #print(output_img_path)
                    img_new = self.noisy(artifacts[i], image)
                    cv2.imwrite(output_img_path, img_new)
                #sys.exit()


    def noisy(self, noise_typ, image):
        if noise_typ == "gauss":
            row, col, ch = image.shape
            mean = 0
            var = 0.5
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return noisy

        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy

        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy

        elif noise_typ == "s&p":
            prob = 0.05
            thres = 1 - prob
            output = np.zeros(image.shape,np.uint8)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rdn = random.random()
                    if rdn < prob:
                        output[i][j] = 0
                    elif rdn > thres:
                        output[i][j] = 255
                    else:
                        output[i][j] = image[i][j]
            return output

def main():
    cwd = os.getcwd()
    iam_ruled_data = cwd + '/dataset/IAM/ruled_data'
    iam_artifacts_data = cwd + '/dataset/IAM/artifacts'
    obj_rule = AdditionalArtifacts(iam_ruled_data, iam_artifacts_data)
    obj_rule.additional_artifacts()

if __name__ == '__main__':
    main()
