from facenet_pytorch import MTCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
import mmcv, cv2
import os
from PIL import Image, ImageDraw
from IPython import display

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

# Playing video from file:
cap = cv2.VideoCapture('spasm2.mp4')

'''try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')'''

currentFrame = 0
frame_loop = 0

frames_tracked = []
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    try:
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except:
        break

    if frame_loop >= 1:
        frame_loop -= 1
        continue

    frame_loop += 2

    # Saves image of the current frame in jpg file
    #name = './data/frame' + str(currentFrame) + '.jpg'
    #print ('Creating...' + name)
    #cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

    print('\rTracking frame: {}'.format(currentFrame + 1), end='')

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    #plt.imshow(frame, aspect="auto")
    #plt.show()

    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)

    try:
        draw.rectangle(boxes[0].tolist(), outline=(255, 0, 0), width=6)
    except:
        pass


    # Add to frame list
    frames_tracked.append(frame_draw.resize((1920, 1080), Image.BILINEAR))
    #plt.imshow(frame_draw, aspect="auto")
    #plt.show()


print('\nDone')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#video = mmcv.VideoReader('video.mp4')
#frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]


"""display.Video('video.mp4', width=640)

frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

    # Add to frame list
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
print('\nDone')"""


d = display.display(frames_tracked[0], display_id=True)
i = 1
"""try:
    while True:
        d.update(frames_tracked[i % len(frames_tracked)])
        i += 1
except KeyboardInterrupt:
    pass"""




dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_tracked = cv2.VideoWriter('video_tracked2.mp4', fourcc, 10, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()

