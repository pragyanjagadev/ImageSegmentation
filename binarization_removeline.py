import sys
import cv2
import numpy as np
from PIL import Image
import os
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
from evaluator import Eval_thread
from eval_dataloader import EvalDataset
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity


class Binarization:
    def __init__(self, iam_augmented_data, iam_removed_rule_data, iam_cropped_data):
        self.input = iam_augmented_data
        self.output = iam_removed_rule_data
        self.output_actual = iam_cropped_data

    def removeRules(self):
        """
           Draw line/rule after cropping the image dynamically and save it
        """
        for subdir, dirs, files in os.walk(self.input):
            for file in files:
                if file != '.DS_Store':
                    # remove lines
                    input_file = self.input + '/' + file
                    output_file = self.output + '/' + file
                    self.lineRemove(input_file, output_file, self.output)
                # exit()

    ###steps -
    """Read a local image.
    Convert the image from one color space to another.
    Apply a fixed-level threshold to each array element.
    Get a structuring element of the specified size and shape for morphological operations.
    Perform advanced morphological transformations.
    Find contours in a binary image.
    Repeat step 4 with different kernel size.
    Repeat step 5 with a new kernel from step 7.
    Show the resultant image."""

    def lineRemove(self, input_file, output_path):
        """
        removing rules from text image using binarization
        :Parameters:
          origimg : path of input image
          output_path : path of output image to be saved
        :Returns:
          None
        """
        self.input_file = input_file
        self.output_path = output_path
        # converting image background
        img = Image.open(self.input_file)
        img = img.convert("RGB")
        datas = img.getdata()
        new_image_data = []
        for item in datas:
            if item[0] in list(range(190, 256)):
                new_image_data.append((255, 255, 255))
            else:
                new_image_data.append(item)

        img.putdata(new_image_data)
        img.save(self.output_path)

        origimg = cv2.imread(self.output_path)

        result = origimg.copy()

        im = Image.open(self.output_path)
        im_color = max(im.getcolors(im.size[0] * im.size[1]))

        gray = cv2.cvtColor(origimg, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print(len(cnts))
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, im_color[1], 5)

        cv2.imwrite(self.output_path, result)
        # print(self.output_path)

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

    def evaulation_metrics(self):
        self.output_pred = self.output

        for subdir, dirs, files in os.walk(self.output_actual):
            for file in files:
                if file != '.DS_Store':
                    output_actual_file = self.output_actual + '/' + file
                    output_pred_file = self.output_pred + '/' + file

                    actual_img = imread(output_actual_file)
                    actual_img = rgb2gray(actual_img)
                    pred_img = imread(output_pred_file)
                    pred_img = rgb2gray(pred_img)

                    actual_img = np.array(Image.fromarray(actual_img.astype(np.uint8)).resize((800, 800)))
                    pred_img = np.array(Image.fromarray(pred_img.astype(np.uint8)).resize((800, 800)))

                    score, diff = structural_similarity(actual_img, pred_img, full=True)
                    print("Similarity Score: {:.3f}%".format(score * 100))

                    # sys.exit()


def main():
    cwd = os.getcwd()
    iam_augmented_data = cwd + '/dataset/IAM/ruled_data'
    iam_removed_rule_data = cwd + '/dataset/IAM/removedrule_4thsept'
    iam_cropped_data = cwd + '/dataset/IAM/cropped_data'
    pred_dir = cwd + '/dataset/IAM/removedrule_4thsept'
    gt_dir = cwd + '/dataset/IAM/cropped_data'
    output_dir = cwd + '/dataset/IAM/output'
    obj_rule = Binarization(iam_augmented_data, iam_removed_rule_data, iam_cropped_data)
    obj_rule.removeRules()

    # similarity score
    obj_rule.evaulation_metrics()
    # other metrices
    threads = []
    loader = EvalDataset(pred_dir, gt_dir)
    thread = Eval_thread(loader, 'method', 'dataset', output_dir, 'cfg.cuda')
    threads.append(thread)

    for thread in threads:
        print(thread.run())


if __name__ == '__main__':
    main()
