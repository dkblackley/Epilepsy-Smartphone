import torch
import torchvision
import cv2
from PIL import Image, ImageDraw
from body_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                           num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

threshold = 0.965

# Playing video from file:
cap = cv2.VideoCapture('spasm2.mp4')

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

    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = frame.copy()
    # transform the image
    frame = transform(frame)
    # add a batch dimension
    frame = frame.unsqueeze(0).to(device)
    masks, boxes, labels = get_outputs(frame, model, threshold)
    #result = draw_segmentation_map(orig_image, masks, boxes, labels)
    # Draw faces
    draw = ImageDraw.Draw(orig_image)
    try:
        for box in boxes:
            box = [box[0][0], box[0][1], box[1][0], box[1][1]]
            draw.rectangle(box, outline=(255, 0, 0), width=6)
    except:
        pass
    # visualize the image
    #plt.imshow(orig_image, aspect="auto")
    #plt.show()
    cv2.waitKey(0)
    # set the save path
    save_path = f"image"
    frames_tracked.append(orig_image.resize((1920, 1080), Image.BILINEAR))


print('\nDone')

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_tracked = cv2.VideoWriter('video_tracked2.mp4', fourcc, 10, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()