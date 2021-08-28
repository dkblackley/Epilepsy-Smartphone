"""
utils.py - holds any helpful utility function for other files
"""

import os
import cv2
import csv
import torch
from PIL import Image
import torch.optim as optim
import numpy as np


from epilepsy_classification.model import Classifier

LABELS = {'INF': 0, 'MYC': 1}
NUMBERS = {0: 'INF', 1: 'MYC'}


def make_labels(path):
    """
    Deprecated method used to strip previous dataset down to binary classifier
    :param path: Path to the labels
    :type path: str
    """
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


def change_into_frames(path_to_data, path_to_save, labels):
    """
    changes videos into just a bunch of images in a folder. Useful for the Pytorch VideoDataloader
    :param path_to_data: path to the videos
    :type path_to_data: str
    :param path_to_save: path to the directory you wish to save the video frames to
    :type path_to_save: str
    :param labels: the list of labels for each video
    :type labels: list
    """

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

            frame.save(path_to_save + real_name + '/img_' + str(frames).zfill(5) + '.jpg')

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
    :param filename: path to the file to read
    :type filename: str
    :param to_num: Whether or not to read file as a string or list of floats
    :type to_num: Bool
    :return: the read csv
    :rtype: list
    """

    list_to_return = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if to_num:
                for i in range(0, len(row)):
                    try:
                        row[i] = float(row[i])
                    except:
                        pass


            list_to_return.append(row)

    return list_to_return


def num_boxes_greater_than_ratio(boxes, debug=False, ratio=0.8):
    """
    check if the number of boxes passed in are greater than the specified threshold
    :param boxes: the bounding boxes
    :type boxes: list
    :param debug: Whether or not to print debug messages
    :type debug: Bool
    :param ratio: the ratio to determine how many boxes are acceptable
    :type ratio: float
    :return: True or False, depending on whether the number of valid boxes passes the threshold
    :rtype: Bool
    """
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
    """
    saves the losses and accuracies of the network, useful for further training
    :param path: path specifying where to save the data
    :type path: str
    :param losses: the list of losses
    :type losses: list
    :param accuracies: the list of accuracies
    :type accuracies: list
    :param loss_filename: the default filename for the loss file
    :type loss_filename: str
    :param accuracry_filname:  the default filename for the accuracy file
    :type accuracry_filname: str
    :param debug: Whether or not to print debug messages
    :type debug: Bool
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    if debug:
        print(f"Saving to {path + loss_filename}.csv")
    write_to_csv(path + f"{loss_filename}.csv", losses)
    write_to_csv(path + f"{accuracry_filname}.csv", accuracies)

def get_results(paths):
    """
    loads in the saved results
    :param paths: paths to the accuracy and loss
    :type paths: list
    :return: the lists of accuracy and loss, returns in the same order you passed in
    :rtype: list
    """
    list = []
    for path in paths:
        list.append(read_from_csv(path, True))

    return list

def save_model(model, optim,  path):
    """
    Saves the Pytorch model and optimizer
    :param model: The Pytorch Classifier
    :type model: nn.Module
    :param optim: The pytorch optimizer
    :type optim: nn.optimizer
    :param path: path to save model to
    :type path: str
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    states = {'network': model.state_dict(),
              'optimizer': optim.state_dict()}

    torch.save(states, path + "/model")

def load_model(path, frame_segments, device):
    """
    Loads model and optimizer from specified path
    :param path: path to model
    :type path: str
    :param frame_segments: number of frame segments to use
    :type frame_segments: int
    :param device: device to load model on to
    :type device: str
    :return: The model and the optimizer
    :rtype: list
    """
    net = Classifier(frame_segments, dropout=0.35)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
    states = torch.load(path, map_location=device)

    net.load_state_dict(states['network'])
    optimizer.load_state_dict(states['optimizer'])

    return net, optimizer


def make_results_LATEX():
    """
    Changes saved results into a format able to be easily pasted into LATEX
    :return: string to be pasted
    :rtype: str
    """

    results = []
    for i in range(0, 10):
        filename = f"models/{i}-FOLD_MODEL/LOSO__RESULTS.csv"
        current = read_from_csv(filename, to_num=True)
        result = []

        for item in current:
            result.append(item[:2])
        result.pop(0)

        results.append(result)
    results = np.array(results)

    string = ""

    for i in range(0, len(results[0])):
        rows = results[:, i:i + 1]
        rows = np.squeeze(rows, 1)

        first = True
        for row in rows:
            if first:
                string += str(row[0]) + " & "
                first = False
            string += str(round(float(row[1]), 3)) + " & "
        string += "\\\\"
        string += '\n'

    return string