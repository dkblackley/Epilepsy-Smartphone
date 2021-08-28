"""
seg_utils.py - Houses the utility functions used by the various body and face detection algorithms
"""

from segmentation.sort import *
import math
import torch

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
    G|iven multiple boxes, returns the one closest to the centre of the image (presuming a 1080x1920 image)
    :param boxes: A list containing the coords of two boxes
    :type boxes: list
    :return: the box closest to the centre and the index identifying which box it was
    :rtype: list
    """
    magnitudes = []
    center = (1080/2, 1920/2)

    for box in boxes:
        x1, y1, width, height = box

        box_center = [(x1 + width)/2, (y1 + height)/2]

        # Calculate the magnitude of the distance between box centre and image centre
        magnitudes.append(math.sqrt((center[0] - box_center[0])**2)+(center[1] - box_center[1])**2)

    magnitudes = np.array(magnitudes)

    index = (np.abs(magnitudes)).argmin()

    return boxes[index], index

def get_outputs(image, model, threshold):
    """
    Used for the body segmentation algorithm, pushes the image through the model
    :param image: image to run through model
    :type image: Tensor
    :param model: model to push the image through
    :type model: nn.Module
    :param threshold: Threshold to determine how confident the model needs to be in its prediction
    :type threshold: float
    :return: Boxes found in the image
    :rtype: ndarray
    """

    with torch.no_grad():
        # forward pass of the image through the model
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])] for i in outputs[0]['boxes'].detach().cpu()]
    boxes = [boxes[i] for i in thresholded_preds_inidices]

    if len(boxes) > 1:
        box, index = find_best_box(boxes)
        score = scores[index]
    elif len(boxes) <= 0:
        return np.empty((0, 5))
    else:
        box = boxes[0]
        score = scores[0]

    box.append(score)
    npbox = np.empty((1, 5))
    npbox[0] = box

    return npbox
