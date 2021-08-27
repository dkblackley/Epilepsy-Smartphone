"""
seg_utils.py - Houses the utility functions used by the various body and face detection algorithms
"""

from segmentation.sort import *
import math

__author__     = ["Daniel Blackley"]
__copyright__  = "Copyright 2021, Epilepsy-Smartphone"
__credits__    = ["Daniel Blackley", "Stephen McKenna", "Emanuele Trucco"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@gmail.com"
__status__     = "Development"


def apply_sort(boxes, mot_tracker):
    """
    Applies the SORT tracking algorithm to a series of bounding boxes
    :param boxes: the series of incoming bounding boxes
    :type boxes: ndarray
    :param mot_tracker: the tracking algorithm to use
    :type mot_tracker: object
    :return: A list of tracked bounding boxes
    :rtype: List
    """
    tracked_object = []

    for box in boxes:
        if np.sum(box) <= 0:
            box = np.empty((0, 5))
        else:
            box = np.array([box])
        tracked = mot_tracker.update(box)

        tracked_object.append(tracked)

    return tracked_object

def find_best_box(boxes):
    """
    G|iven two boxes, returns the one closest to the centre of the image (presuming a 1080x1920 image)
    :param boxes: A list containing the coords of two boxes
    :type boxes: list
    :return: the box closest to the centre and the index identifying which box it was
    :rtype: list
    """
    #TODO update this
    magnitudes = []
    center = (1920/2, 1080/2)

    for box in boxes:
        x1, y1, x2, y2 = box

        box_center = [(x1 + x2)/2, (y1 + y2)/2]

        magnitudes.append(math.sqrt((center[0] - box_center[0])**2)+(center[1] - box_center[1])**2)

    magnitudes = np.array(magnitudes)

    index = (np.abs(magnitudes)).argmin()

    return boxes[index], index


