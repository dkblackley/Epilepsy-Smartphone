from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import mmcv, cv2
import os
from PIL import Image, ImageDraw
from IPython import display
from segmentation.seg_utils import *


def find_face(device, frames, display=False, debug=False, threshold=0.96, save=False):

    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)


    currentFrame = 0
    # frame_loop = 0
    tracking_boxes = np.zeros(shape=(len(frames), 5))

    frames_tracked = []
    for frame in frames:

        currentFrame += 1

        if debug:
            print('\rTracking frame: {}'.format(currentFrame + 1), end='')

        # Detect and track faces
        boxes, confidence = mtcnn.detect(frame)


        thresholded_preds_inidices = [confidence.tolist().index(i) for i in confidence if i > threshold]

        boxes = [boxes[i] for i in thresholded_preds_inidices]

        if boxes is not None:

            if len(boxes) > 1:
                box, index = find_best_box(boxes)
            else:
                index = 0

            temp = np.append(boxes[index], confidence[index]).tolist()
            box = np.array(temp)
            tracking_boxes[currentFrame - 1] = box.copy()

        if display and debug:
            plt.imshow(frame, aspect="auto")
            plt.show()

        """# Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)

        try:
            draw.rectangle(boxes[0].tolist(), outline=(255, 0, 0), width=6)
        except:
            pass

        # Add to frame list
        frames_tracked.append(frame_draw.resize((1920, 1080), Image.BILINEAR))
        # plt.imshow(frame_draw, aspect="auto")
        # plt.show()"""

    return tracking_boxes

    # print('\nDone')



