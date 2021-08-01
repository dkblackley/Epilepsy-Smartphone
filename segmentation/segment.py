import torch
from PIL import Image, ImageDraw
import mmcv, cv2
import body_detect
import face_detect
from seg_utils import *
from sort import *

def return_ROI(frames):
    pass

def make_video(video):
    pass


def find_boxes(video, track, segmentation, batch_size, save=False):

    frame_list = []
    frame_loop = 0
    all_boxes = []
    if track:
        mot_tracker = Sort()

    while (True):

        # Capture frame-by-frame
        ret, frame = video.read()
        batch = torch.empty(0)

        try:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except:

            break

        frame_list.append(frame)

        if frame_loop % batch_size == 0:

            if segmentation == "body":
                boxes = body_detect.detect_body("cpu", frame_list, display=False, debug=False, save=False)
            else:
                boxes = face_detect.find_face("cpu", frame_list, display=False, debug=False, save=False)

            if track:

                img_size = 224

                pad_x = max(frame.shape[0] - frame.shape[1], 0) * (img_size / max(frame.shape))
                pad_y = max(frame.shape[1] - frame.shape[0], 0) * (img_size / max(frame.shape))
                unpad_h = img_size - pad_y
                unpad_w = img_size - pad_x

                padding = [unpad_h, unpad_w, pad_x, pad_y]

                boxes = apply_sort(boxes, mot_tracker, padding, frame)

            all_boxes.append(boxes)

            frame_list.clear()

        frame_loop += 1
