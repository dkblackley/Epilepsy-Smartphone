"""
face_detect.py - The file responsible for finding the patients face and return a bounding box around it.
"""

from facenet_pytorch import MTCNN
from segmentation.seg_utils import *

__author__     = ["Daniel Blackley"]
__copyright__  = "Copyright 2021, Epilepsy-Smartphone"
__credits__    = ["Daniel Blackley", "Stephen McKenna", "Emanuele Trucco"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@gmail.com"
__status__     = "Development"

def find_face(device, frames, display=False, debug=False, threshold=0.7):
    """
    Uses Multi-task Cascaded Convolutional Networks (MTCNN) to find bounding boxes around a given list of images. If
    it can't find an image it adds a a list of 4 0s
    :param device: device to run model on
    :type device: str
    :param frames: list of PIL images
    :type frames: list
    :param display: Whether or not to display the faces with the chosen bounding box. Useful for debugging
    :type display: Bool
    :param debug: Whether or not to display debug messages
    :type debug: Bool
    :param threshold: MTCNN gives a level of confidence with its boxes, this is the threshold to decide what
    face is acceptable
    :type threshold: float
    :return: ndarray of bounding boxes
    :rtype: ndarray
    """

    model = MTCNN(device=device, margin=20, min_face_size=40)
    currentFrame = 0
    tracking_boxes = np.zeros(shape=(len(frames), 5))

    for frame in frames:
        currentFrame += 1

        if debug:
            print('\rTracking frame: {}'.format(currentFrame + 1), end='')

        # Detect face
        boxes, confidence = model.detect(frame)

        if boxes is None:
            tracking_boxes[currentFrame - 1] = np.zeros(5)
            continue

        thresholded_preds_inidices = [confidence.tolist().index(i) for i in confidence if i > threshold]
        boxes = [boxes[i] for i in thresholded_preds_inidices]

        if boxes:
            if len(boxes) > 1:
                box, index = find_best_box(boxes)
            else:
                index = 0

            temp = np.append(boxes[index], confidence[index]).tolist()
            box = np.array(temp)
            tracking_boxes[currentFrame - 1] = box.copy()
        else:
            tracking_boxes[currentFrame - 1] = np.zeros(5)
            continue

        if display and debug:
            plt.imshow(frame, aspect="auto")
            plt.show()

    return tracking_boxes




