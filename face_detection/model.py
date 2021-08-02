from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import mmcv, cv2
import os
from PIL import Image, ImageDraw
from IPython import display

def find_face(device, frames, display=False, debug=False, save=False):

    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)

    """# Playing video from file:
    cap = cv2.VideoCapture('spasm2.mp4')"""

    currentFrame = 0
    # frame_loop = 0

    frames_tracked = []
    while (True):
        """# Capture frame-by-frame
        ret, frame = cap.read()

        try:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except:
            break"""

        """if frame_loop >= 1:
            frame_loop -= 1
            continue

        frame_loop += 2"""


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

    print('\nDone')

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    d = display.display(frames_tracked[0], display_id=True)
    i = 1
    """try:
        while True:
            d.update(frames_tracked[i % len(frames_tracked)])
            i += 1
    except KeyboardInterrupt:
        pass"""

"""dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_tracked = cv2.VideoWriter('video_tracked2.mp4', fourcc, 10, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()"""

