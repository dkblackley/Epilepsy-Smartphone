"""
segment.py - File responsible for setting up and saving the various ROI segmentation algorithms. If it can't detect a
box ,-[1,-1,-1,-1] is saved instead
"""
from PIL import ImageDraw, Image
import segmentation.body_detect as body_detect
import segmentation.face_detect as face_detect
import utils
from segmentation.seg_utils import *
from segmentation.sort import *
from tqdm import tqdm
import cv2



def make_video(frames_tracked, path):
    """
    makes a video given a list of frames
    :param frames_tracked: list of frames to make a video from
    :type frames_tracked: list
    :param path: location to save video to
    :type path: str
    """
    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video_tracked = cv2.VideoWriter(path, fourcc, 10, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()


def set_up_boxes(path, device):
    """
    Entrypoint for the segmentation file. Cycles through all the videos in the specified path and attempts to set up
    the bounding boxes for that video
    :param path: path to the videos
    :type path: str
    :param device: device to run models on
    :type device: str
    """

    for filename in tqdm(os.listdir(path)):
        # If not a video
        if filename[-4:] != '.mp4':
            continue

        print("\nGenerating boxes for " + filename)
        new_filename = filename[:-4]
        video = cv2.VideoCapture(path + filename)
        find_boxes(video, 'face', 12, track=True, csv=path + new_filename + '_face_boxes.csv', device=device)

        video = cv2.VideoCapture(path + filename)
        find_boxes(video, 'body', 12, track=True, csv=path + new_filename + '_body_boxes.csv', device=device)



def find_boxes(video, segmentation, batch_size, device='cpu', track=True, save='', csv=''):
    """
    Main function for cycling through a video and extracting the specified region of interest
    :param video: The video to cycle through
    :type video: VideoCapture
    :param segmentation: The region we want to draw boxes around, can be face or body
    :type segmentation: str
    :param batch_size: batch size to use
    :type batch_size: int
    :param track: Whether we should apply the SORT tracking algorithm to boxes
    :type track: Bool
    :param save: If we should save the video with bounding box added
    :type save: Bool
    :param csv: Where to save the bounding box to
    :type csv: str
    :return The list of bounding boxes
    :rtype list
    """

    frame_list = []
    frame_loop = 0
    all_boxes = []
    end = False
    if track:
        mot_tracker = Sort()

    if save:
        frames_tracked = []

    while (True):
        # Capture frame-by-frame
        ret, frame = video.read()
        frame_loop += 1

        try:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_list.append(frame)
        except:
            if not end:
                end = True
                frame_loop = batch_size
            else:
                break

        if frame_loop % batch_size == 0:

            if segmentation == "body":
                boxes = body_detect.detect_body(device, frame_list, debug=False)
            else:
                boxes = face_detect.find_face(device, frame_list, display=False, debug=False)

            if track:
                boxes = apply_sort(boxes, mot_tracker)

            else:
                new_box = []
                for box in boxes:
                    new_box.append(np.array([box]))
                boxes = new_box

            all_boxes.append(boxes)

            if save:
                for i in range(0, len(frame_list)):
                    # Draw faces
                    frame = frame_list[i]
                    frame_draw = frame.copy()
                    draw = ImageDraw.Draw(frame_draw)

                    try:
                        box = boxes[i][0].tolist()
                        box.pop()
                        draw.rectangle(box, outline=(255, 0, 0), width=6)
                    except:
                        pass
                    # Add to frame list
                    frames_tracked.append(frame_draw.resize((1920, 1080), Image.BILINEAR))

                    """frame_draw.show()
                    frame = frame.crop(box)
                    frame.show()"""

            frame_list = []

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()

    if csv:
        new_boxes = []
        for boxes in all_boxes:
            for box in boxes:
                if box.sum() == 0:
                    box = [-1, -1, -1, -1]
                else:
                    box = box[0].tolist()
                    box.pop()
                new_boxes.append(box)
        utils.write_to_csv(csv, new_boxes)

    if save:
        make_video(frames_tracked, save)

    return all_boxes