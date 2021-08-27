"""
body_detect.py - The file responsible for finding the patients body and return a bounding box around it
"""

import torchvision
from segmentation.seg_utils import get_outputs
from torchvision.transforms import transforms as transforms
import numpy as np

__author__     = ["Daniel Blackley"]
__copyright__  = "Copyright 2021, Epilepsy-Smartphone"
__credits__    = ["Daniel Blackley", "Stephen McKenna", "Emanuele Trucco"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@gmail.com"
__status__     = "Development"


def detect_body(device, frames, debug=False, threshold=0.9):
    """
    given a list of images, uses a pretrained Keypoint RCNN with a resnet50 backbone  to identify a series of bounding
    boxes where the body is. If it can't find an image it adds a a list of 4 0s
    :param device: the device to train the model on. GPU highly recommended
    :type device: str
    :param frames: list of PIL images
    :type frames: list
    :param debug: Whether debug messages should be printed
    :type debug: Bool
    :param threshold: the RCNN returns a measure of confidence with box, this determines how confident the model needs
    to be
    :type threshold: float
    :return: ndarray of bounding boxes relating to given frames
    :rtype: ndarray
    """

    # initialize the model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    # set the computation device
    device = device
    # load the model on to the computation device and set to eval mode
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    currentFrame = 0
    boxes=[]

    if debug:
        print('\rTracking frame: {}'.format(currentFrame + 1), end='')

    for frame in frames:
        frame = transform(frame)
        box = get_outputs(frame.unsqueeze(0), model, threshold)
        boxes.append(box)

    npboxes = np.empty((len(boxes), 5))

    for i in range(0, len(boxes)):
        if boxes[i].sum() > 0:
            npboxes[i] = boxes[i]
        else:
            npboxes[i] = np.zeros(5)

    return npboxes