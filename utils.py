import os
import cv2
import csv
import torch
import numpy as np
from PIL import Image, ImageDraw
from epilepsy_classification.model import Classifier
import torch.optim as optim

LABELS = {'INF': 0, 'MYC': 1}
NUMBERS= {0: 'INF', 1: 'MYC'}

def make_labels(path):

    labels = [['video', 'chew', 'clap']]

    for direct in os.listdir(path):
        if os.path.isdir(path + direct):
            cur_path = path + direct

            for filename in os.listdir(cur_path):
                list = [filename]
                if 'chew' in direct:
                    list.append('1')
                    list.append('0')
                else:
                    list.append('0')
                    list.append('1')
                labels.append(list)
        else:
            continue

    write_to_csv(path + 'labels.csv', labels)

def change_videos_fps(path_to_data):

    filenames = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]

    for filename in filenames:

        if filename[-4:] != '.mp4':
            continue

        filename = "spasm.mp4"
        new_frames = []
        frame_loop = 0

        filename = path_to_data + filename

        # Playing video from file:
        cap = cv2.VideoCapture(filename)

        while(True):

            # Capture frame-by-frame
            ret, frame = cap.read()

            if frame_loop >= 1:
                frame_loop -= 1
                continue

            try:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except:
                break

            new_frames.append(frame)
            frame_loop += 2

        dim = new_frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        video_tracked = cv2.VideoWriter(filename, fourcc, 10, dim)

        for frame in new_frames:
            video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video_tracked.release()
        cap.release()

def change_into_frames(path_to_data, labels):

    #filenames = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    labels.pop(0)

    for i in range(0, len(labels)):
        ID = labels[i].index('1') - 1
        labels[i] = [labels[i][0], ID]

    annotations = []

    for filename, label in labels:

        frames = 0

        if filename[-4:] != '.mp4':
            continue

        real_name = filename[:-4]
        filename = path_to_data + filename

        if not os.path.isdir(path_to_data + real_name):
            os.mkdir(path_to_data + real_name)

        # Playing video from file:
        cap = cv2.VideoCapture(filename)

        frame_loop = 0

        while (True):

            # Capture frame-by-frame
            ret, frame = cap.read()

            if frame_loop >= 1:
                frame_loop -= 1
                continue

            try:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except:
                break

            frame_loop += 2
            frames += 1

            frame.save(path_to_data + real_name + '/img_' + str(frames).zfill(5) + '.jpg')

        annotations.append([real_name + ' 1' + f' {frames} ' + str(label)])
        cap.release()

    write_to_csv('datasets/annotations.txt', annotations)


def write_to_csv(path, list_to_write):
    """
    writes a list of items to a output file.
    :param list_to_write: list to write to csv, expects as a list of lists, each list is a new line
    :param filename: location and name of csv file
    """
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list_to_write)

def read_from_csv(filename, to_num=False):
    """
    read a file of comma separated values
    :return: a list of the read values
    """

    list_to_return = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for i in range(0, len(row)):
                row[i] = float(row[i])


            list_to_return.append(row)

    return list_to_return

def num_boxes_greater_than_ratio(boxes, debug=False, ratio=0.8):

    total = len(boxes)
    invalid = 0
    valid = 0

    for box in boxes:
        if sum(box) > 0:
            valid += 1
        else:
            invalid += 1

    if debug:
        print(f"{(valid/total)*100}% of boxes are valid")
        print(f"{(invalid/total)*100}% of boxes are invalid\n")

    if valid/total > ratio:
        return True
    else:
        return False

def save_results(path, losses, accuracies, loss_filename="losses", accuracry_filname="accuracy", debug=False):

    if not os.path.isdir(path):
        os.mkdir(path)
    if debug:
        print(f"Saving to {path + loss_filename}.csv")
    write_to_csv(path + f"{loss_filename}.csv", losses)
    write_to_csv(path + f"{accuracry_filname}.csv", accuracies)

def get_results(paths):

    list = []
    for path in paths:
        list.append(read_from_csv(path, True))

    return list

def save_model(model, optim,  path):
    if not os.path.isdir(path):
        os.makedirs(path)

    states = {'network': model.state_dict(),
              'optimizer': optim.state_dict()}

    torch.save(states, path + "/model")

def load_model(path, frame_segments, device):

    net = Classifier(frame_segments, dropout=0.35)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
    states = torch.load(path, map_location=device)

    net.load_state_dict(states['network'])
    optimizer.load_state_dict(states['optimizer'])

    return net, optimizer





