from segmentation.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
import cv2
import numpy as np
import random
import torch
from PIL import Image

from segmentation.sort import *
import math

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def apply_pad(box, padding, number):

    for i in range(0, int(len(padding)/2)):
        box[i] -= padding[i]
        box[i] = round(box[i]/number)*number

    for i in range(0, int(len(padding)/2)):
        box[i+2] += padding[i+2]
        box[i+2] = round(box[i+2]/number)*number


def apply_sort(boxes, mot_tracker, img, round, pad=False):

    tracked_object = []

    for box in boxes:
        if np.sum(box) <= 0:
            box = np.empty((0, 5))
        else:
            box = np.array([box])
        #tracked_object[temp] = mot_tracker.update(box)
        tracked = mot_tracker.update(box)

        if pad:
            try:
                apply_pad(tracked[0], pad, round)
            except:
                pass


        tracked_object.append(tracked) #TODO remove this extra bracket

    #tracked_object = np.array(tracked_object)

    """tracked = []
    for tracked_object in tracked_objects:
        x1, y1, x2, y2, obj_id = tracked_object
        tracked.append([x1, y1, x2, y2])"""


    """unpad_h, unpad_w, pad_x, pad_y = padding

    unique_labels = boxes[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])"""

    return tracked_object

    """color = colors[int(obj_id) % len(colors)]
        color = [i * 255 for i in color]
        cls = classes[int(cls_pred)]
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1)
        cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)"""

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the model
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    #masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    #masks = masks[:thresholded_preds_count]
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

def find_best_box(boxes):

    magnitudes = []
    center = (1920/2, 1080/2)

    for box in boxes:
        x1, y1, x2, y2 = box

        box_center = [(x1 + x2)/2, (y1 + y2)/2]

        magnitudes.append(math.sqrt((center[0] - box_center[0])**2)+(center[1] - box_center[1])**2)

    magnitudes = np.array(magnitudes)

    index = (np.abs(magnitudes)).argmin()


    return boxes[index], index

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1
    beta = 0.6  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        # convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=cv2.LINE_AA)

    return image

