"""
File used for handling the data set
"""

from __future__ import print_function, division
import os

import numpy
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TrsF
from tqdm import tqdm
import torch
import cv2

import utils


class data_set(Dataset):
    """
    class responsible for handling and dynamically retreiving data from the data set
    """
    def __init__(self, path_to_data, train_transforms, test_transforms, labels_path, segmentation=None):
        """
        Init responsible for holding the list of filenames from which you can fetch data from
        :param root_dir: path to the video files
        :param labels_path: path to the filenames and labels
        :param transforms: transforms to be applied to the data
        """

        self.root_dir = path_to_data
        self.file_names = os.listdir(self.root_dir)
        self.file_names.sort()
        self.file_names.pop(0)
        self.tr_transforms = train_transforms
        self.te_transforms = test_transforms
        self.segmentation = segmentation

        self.labels = pd.read_csv(labels_path)
        self.classes = self.labels.columns[1:3].values
        self.labels = np.array(self.labels.values.tolist())


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        Dynamically loads and returns an image at specified index with label attached.
        If there is no label then it returns False as a label
        :param index: index of image to load
        :return: dictionary containing image and label
        """

        if self.labels is False:
            file_name = self.file_names[index]
        else:
            file_name = self.labels[index][0]
        full_path = os.path.join(self.root_dir, file_name)
        # Playing video from file:
        video = cv2.VideoCapture(full_path)

        face_boxes = utils.read_from_csv((full_path[:-4] + '_face_boxes.csv'), to_num=True)
        body_boxes = utils.read_from_csv((full_path[:-4] + '_body_boxes.csv'), to_num=True)

        if self.labels is False:
            label = False
        else:
            #label = self.get_class_name(self.labels[index][1:])
            label = self.labels[index][1:].astype(np.float)
            label = np.argmax(label)
            label = torch.tensor(float(label))

        data = {'video': video, "label": label, 'filename': file_name, 'face': face_boxes, 'body': body_boxes}

        return data

    def get_filename(self, index):
        return self.labels[index][1:] + '.mp4'

    def get_label(self, index):
        """
        Returns the label as an integer at the specified index
        """
        return self.get_class_name(self.labels[index][1:-1])

    def get_all_labels(self, test_indexes):

        answers = []
        for i in range(0, len(test_indexes)):
            answers.append(self.get_class_name(self.labels[test_indexes[i]][1:-1]))

        return answers


    def get_class_name(self, numbers):
        """
        returns the class names based on the number, i.e. 0 = MEL, 1 = NV... etc.
        :param numbers: a list of values, where one number is 1.0 to represent the class
        :return:
        """
        index = np.where(numbers == '1')
        return index[0][0]

    def add_transforms(self, transforms):
        self.transforms = transforms


class RandomCrop(object):
    # Unused
    """
    Class used to randomly crop the image
    """

    def __init__(self, output_size):
        """
        Init takes in the desired crop size
        :param output_size:
        """

        # If output size is one parameter, make it a tuple of two of the same parameters
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        """
        Randomly crops off the side of the image to the specified size
        :param image: Image to be cropped
        :return: the cropped image
        """

        image = np.asarray(image)
        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        image = image[top: top + new_height, left: left + new_width]

        image = Image.fromarray(image, 'RGB')

        return image


class RemoveBorders(object):

    def __init__(self, image_size, tolerance=0):
        self.tol = tolerance
        self.image_size = image_size
        self.index = 0

    def __call__(self, img):
        """
        Taken and adapted from https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
        :param image: Image to be cropped
        :return: the image with black borders removed
        """

        img.show()

        image = np.asarray(img)
        self.index = self.index + 1

        # Delete last 5 rows and columns
        image = np.delete(image, np.s_[self.image_size - 5:], 1)
        image = np.delete(image, np.s_[self.image_size - 5:], 0)

        image = np.delete(image, np.s_[:5], 1)
        image = np.delete(image, np.s_[:5], 0)

        prev_size = np.sum(image)

        mask = image > self.tol
        if image.ndim == 3:
            mask = mask.all(2)
        mask0, mask1 = mask.any(0), mask.any(1)
        image = image[np.ix_(mask0, mask1)]

        new_size = np.sum(image)

        image = Image.fromarray(image, 'RGB')

        image.show()

        if prev_size == new_size:
            return img
        else:
            return image
