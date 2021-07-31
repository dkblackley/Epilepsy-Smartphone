from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import mmcv, cv2
import os
from PIL import Image, ImageDraw
from IPython import display


def return_ROI(frame_batch):
    pass


def find_face(device, frames, display=False, debug=False, save=False):

    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)


    currentFrame = 0
    # frame_loop = 0

    frames_tracked = []
    for frame in frames:

        currentFrame += 1

        if debug:
            print('\rTracking frame: {}'.format(currentFrame + 1), end='')

        # Detect and track faces
        boxes, _ = mtcnn.detect(frame)

        if display and debug:
            plt.imshow(frame, aspect="auto")
            plt.show()

        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)

        try:
            draw.rectangle(boxes[0].tolist(), outline=(255, 0, 0), width=6)
        except:
            pass

        # Add to frame list
        frames_tracked.append(frame_draw.resize((1920, 1080), Image.BILINEAR))
        # plt.imshow(frame_draw, aspect="auto")
        # plt.show()

    # print('\nDone')



