import cv2
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import os
import sys
import argparse
import numpy as np
import random


class ImageAugment:

    def __init__(self, iam_data_path, cropped_data_path, iam_ruled_data_path, iam_augmented_data):

        self.iam_data_path = iam_data_path
        self.cropped_data_path = cropped_data_path
        self.iam_ruled_data_path = iam_ruled_data_path
        self.iam_augmented_path = iam_augmented_data

        ## Define control variables and parse user inputs
        parser = argparse.ArgumentParser()
        parser.add_argument('--numaugs', help='Number of augmented images to create from each original image',
                            default=5)

        args, unknown = parser.parse_known_args()

        #### Define augmentation sequence ####
        self.seq = iaa.Sequential([
            #iaa.ElasticTransformation(alpha=(0.0, 70.0), sigma=5.0),
            iaa.ElasticTransformation(alpha=(0.0, 10.0), sigma=5.0),
            iaa.Fliplr(0.5),  # Horizontal flip 50% of images
            iaa.Crop(percent=(0, 0.10)),  # Crop all images between 0% to 10%
            iaa.GaussianBlur(sigma=(0, 1)),  # Add slight blur to images
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Resize image
                rotate=(-5, 5),  # Rotate image
            )
        ])

        self.num_aug_images = int(args.numaugs)

    def augment(self, img):

        self.img = img
        kernel = np.ones((5, 5), np.uint8)
        erosion = random.choice([False, True])
        dilation = random.choice([False, True])

        if erosion:
            img = cv2.erode(self.img, kernel, iterations=1)

        #if dilation:
            #img = cv2.dilate(self.img, kernel, iterations=1)

        return img

    def createAugmentedImage(self):
        for subdir, dirs, images in os.walk(self.iam_ruled_data_path):
            for img in images:
                input_img = self.iam_ruled_data_path + '/' + img
                output_dir = self.iam_augmented_path + '/'

                # ---- Load image ----#
                img1_bgr = cv2.imread(input_img)  # Load image with OpenCV
                img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)  # Re-color to RGB from BGR

                # ---- Augment image N times----#
                for i in range(self.num_aug_images):
                    img_aug_1 = self.seq(images=[img1])[0]  # Apply augmentation to image

                    img_aug_2 = self.augment(img_aug_1)

                    base_fn = img.replace('.png', '')  # Get image base filename
                    img_aug_fn = base_fn + ('_aug%d' % (i + 1)) + '.png'  # Append "aug#" to filename
                    img_aug_fn1 = output_dir + img_aug_fn
                    img_aug_bgr1 = cv2.cvtColor(img_aug_2, cv2.COLOR_RGB2BGR)  # Re-color to BGR from RGB
                    cv2.imwrite(img_aug_fn1, img_aug_bgr1)  # Save image to disk

                #exit()

def main():
    cwd = os.getcwd()
    iam_data = cwd + '/dataset/IAM/formsI-Z'
    cropped_data = cwd + '/dataset/IAM/cropped_data'
    iam_ruled_data = cwd + '/dataset/IAM/ruled_data'
    iam_augmented_data = cwd + '/dataset/IAM/augmented_data_4sept'
    obj_rule = ImageAugment(iam_data, cropped_data, iam_ruled_data, iam_augmented_data )
    obj_rule.createAugmentedImage()


if __name__ == '__main__':
    main()
