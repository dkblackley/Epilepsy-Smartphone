
"""
training.py - File used for the training and testing of model.py
"""

from tqdm import tqdm
import random
import cv2
import torch.optim as optim
from PIL import Image
import epilepsy_classification.model as model
import torch.nn as nn
import utils
import torch

__author__     = ["Daniel Blackley"]
__copyright__  = "Copyright 2021, Epilepsy-Smartphone"
__credits__    = ["Daniel Blackley", "Stephen McKenna", "Emanuele Trucco"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@gmail.com"
__status__     = "Development"


class Trainer:

    def __init__(self, dataset, frame_segments, tr_transforms, te_transforms, net=False, segment='',
                 save_dir="models/", optimizer=False, threshold=0.75, early_stop=True, device='cpu', save_model=True):
        """
        Init method, deals with loading model and placing it on the correct device
        :param dataset: Path to where the data can be found
        :type dataset: str
        :param frame_segments: the number of frames to load from the video
        :type frame_segments: int
        :param tr_transforms: Transforms to apply to the training images
        :type tr_transforms: Pytorch Compose object
        :param te_transforms: Transforms to apply to the testing imagess
        :type te_transforms: Pytorch Compose object
        :param net: Network to use
        :type net: Classifier
        :param segment: The ROI to segment, can be either 'face', 'body' or the default, which is none.
        :type segment: str
        :param save_dir: Where to save the models during training
        :type save_dir: str
        :param optimizer: The optimizer, if loading a model
        :type optimizer: Pytorch Optimiser
        :param threshold: The threshold to decide on how many boxes are required before a video is discarded, i.e. a
        value of 0.8 means that the videos required boxes for 80% of the video
        :type threshold: float
        :param early_stop: Whether or not to save the lowest loss model separately
        :type early_stop: Bool
        :param device: Device to store model on
        :type device: str
        :param save_model: Whether the models should be saved after very epoch. Can take a lot of space when running
         LOSO
        :type save_model: Bool
        """
        self.dataset = dataset
        self.frame_seg = frame_segments
        self.criterion = nn.BCEWithLogitsLoss()
        self.weight = torch.tensor([1.0, 1.0]).to(device)
        self.tr_transforms = tr_transforms
        self.te_transforms = te_transforms
        self.segment = segment
        self.save_dir = save_dir
        self.threshold = threshold
        self.early_stop = early_stop
        self.device = device
        self.save_model = save_model

        if not net:
            self.net = model.Classifier(frame_segments, dropout=0.35, device=device)
            self.net = self.net.to(device)
            self.optim = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)
        else:
            self.net = net
            self.net = self.net.to(device)
            self.optim = optimizer

    def set_save_dir(self, new_dir):
        """
        setter for the save directory, used in LOSO
        :param new_dir: Directory to change current save dir to
        :type new_dir: str
        """
        self.save_dir = new_dir

    def set_weights(self, indices):
        """
        Sets the weights for the BCELossWithLogits function. using a positive weighting scheme
        :param indices: the indices that are being including in this dataset
        :type indices: int
        """

        x1 = 0
        x2 = 0
        for i in indices:
            data = self.dataset[i]
            label = data['label']

            if label == 0:
                x1 += 1
            else:
                x2 += 1

        self.weight = torch.tensor([x1/x2]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.weight)


    def LOSO(self, epochs, debug=False):
        """
        Removes only a single video for validation while training on the remaining videos. Also known as LOSO validation
        :param epochs: number of epochs to train for
        :type epochs: 7
        :param debug: Whether or not to print debug messages
        :type debug: Bool
        """

        indices = [i for i in range(0, len(self.dataset))]
        i = 0

        if debug:
            print(f"{len(indices)} number of videos before purge")

        while i < len(indices):
            data = self.dataset[indices[i]]

            # mimic7 doesn't work with face segmentation, heck if I know why.
            if self.segment == 'face' and data['filename'] == 'mimic7.mp4':
                if debug:
                    print(f"{data['filename']} Has been removed\n")
                indices.pop(i)

            # Remove videos without enough boxes
            elif self.segment and not utils.num_boxes_greater_than_ratio(data[self.segment], ratio=self.threshold,
                                                                         debug=debug):
                if debug:
                    print(f"{data['filename']} Has been removed\n")
                indices.pop(i)

            # If there are less valid boxes than number of frames we want to use, remove that video
            elif self.segment:
                boxes = data[self.segment]
                num_valid = 0
                for box in boxes:
                    if sum(box) > 0:
                        num_valid += 1

                if num_valid < self.frame_seg:
                    if debug:
                        print(f"Skipping video {data['filename']} due to unsatisfactory bounding boxes\n")
                    indices.pop(i)
                else:
                    i = i + 1
            else:
                i = i + 1

        if debug:
            print(f"{len(indices)} remain after purge")

        results = [["filename", "MIMIC", "INF"]]
        old_dir = self.save_dir

        # Cycle through videos, performing LOSO
        for i in range(0, len(indices)):
            self.save_dir = old_dir + f"LOSO_{self.dataset[indices[i]]['filename'][:-4]}_{self.segment}/"

            if debug:
                print(f"Working on model LOSO_model_{self.dataset[indices[i]]['filename'][:-4]}_{self.segment}")

            self.net = model.Classifier(self.frame_seg, dropout=0.35, device=self.device)
            self.net = self.net.to(self.device)
            self.optim = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)
            indices_copy = indices.copy()
            self.train(epochs, train=indices_copy[:i] + indices_copy[i+1:], val=[indices[i]], debug=debug)

            results.append(self.test([indices[i]], debug)[0])

        utils.write_to_csv(old_dir + f"LOSO_{self.segment}_RESULTS.csv", results)



    def train(self, epochs, shuffle=True, train=None, val=None, split=0.25, debug=False):
        """
        Method responsible for training the model on the training data
        :param epochs: Number of epochs to run for
        :type epochs: int
        :param shuffle: Whether we should shuffle the data
        :type shuffle: Bool
        :param train: Can manually specify which indices to train on, makes split variable useless
        :type train: list
        :param val: Can manually specify which indices to validate on, makes split variable useless
        :type val: list
        :param split: The train/test split. A value of 0.25 means 75% of videos will be used to train the model and
        25% will be used to validate the model
        :type split: float
        :param debug: Whether or not debug messages should be printed
        :type debug: Bool
        """

        t_overall_accuracy = []
        v_overall_accuracy = []
        t_overall_loss = []
        v_overall_loss = []

        # Set the train indices if not specified
        if train is None:
            indices_original = [i for i in range(0, len(self.dataset))]
            train = indices_original[int(len(indices_original) * split):].copy()
            val = indices_original[:int(len(indices_original) * split)].copy()

        # Weight our loss function based on given train indices
        self.set_weights(train)

        for i in range(0, epochs):

            print(f"EPOCH {i} of {epochs}")

            if shuffle:
                random.shuffle(val)
                random.shuffle(train)

            accuracies = []
            losses = []
            self.net.train()
            for current in tqdm(train):

                data = self.dataset[current]
                self.net.reset_states()
                output = -1

                # If unlucky, we can cut out more videos without bounding boxes than frame segments specified,
                # if this occurs try again.
                while(output == -1):
                    output = self.run_through(data, True, debug=debug)

                answer, loss = output
                accuracies.append(answer)
                losses.append(loss)

            if debug:
                print(losses)
                print(accuracies)

            t_overall_loss.append(sum(losses)/len(losses))
            t_overall_accuracy.append(sum(accuracies)/len(accuracies))

            accuracies = []
            losses = []
            self.net.eval()
            for current in tqdm(val):

                data = self.dataset[current]
                self.net.reset_states(requires_grad=False)

                with torch.no_grad():

                    output = -1
                    while (output == -1):
                        output = self.run_through(data, False, average_res=True, debug=debug)

                    answer, loss = output

                accuracies.append(answer)
                losses.append(loss)

            if debug:
                print(losses)
                print(accuracies)

            v_overall_loss.append(sum(losses) / len(losses))
            v_overall_accuracy.append(sum(accuracies) / len(accuracies))

            # Save best model
            if (sum(losses)/len(losses)) <= min(v_overall_loss) and self.early_stop:
                if self.save_model:
                    utils.save_model(self.net, self.optim, self.save_dir + "best_loss/")

                utils.save_results(self.save_dir + "best_loss/", [v_overall_loss], [v_overall_accuracy],
                                   loss_filename="val_losses", accuracry_filname="val_accuracy")
                utils.save_results(self.save_dir + "best_loss/", [t_overall_loss], [t_overall_accuracy])

            if self.save_model:
                utils.save_model(self.net, self.optim, self.save_dir + "model/")
            utils.save_results(self.save_dir + "model/", [v_overall_loss], [v_overall_accuracy],
                               loss_filename="val_losses", accuracry_filname="val_accuracy", debug=debug)
            utils.save_results(self.save_dir + "model/", [t_overall_loss], [t_overall_accuracy], debug=debug)

            print('\nTrain accuracies:')
            print(t_overall_accuracy)
            print('Train losses:')
            print(t_overall_loss)
            print('\nval accuracies:')
            print(v_overall_accuracy)
            print('val losses:')
            print(v_overall_loss)

    def test(self, indices, debug):
        """

        :param indices:
        :type indices:
        :param debug:
        :type debug:
        :return:
        :rtype:
        """
        self.net.eval()
        results = []
        answers = []

        for index in indices:
            data = self.dataset[index]
            self.net.reset_states(requires_grad=False)

            with torch.no_grad():
                answer, _ = self.run_through(data, False, average_res=True, probs=True, debug=debug)

            answers.append([1-answer, answer])
        for answer in answers:
            result = [data['filename'], answer[0], answer[1]]
            results.append(result)

        if debug:
            print(f"\n{data['filename']} result: {result} \n")

        return results


    def run_through(self, sample, train, debug=False, probs=False, average_res=False):
        """
        Run the specified number of frames through the video. The frames taken are taken at random
        :param sample: video, label and filename of video
        :type sample: dict
        :param train: Whether or not we're in train mode, if False go through the entire video
        :type train: Bool
        :param debug: Whether we should print output messages
        :type debug: Bool
        :param probs: Whether we should return a simple 1 or 0 for correct and incorrect or the output probability
        directly
        :type probs: Bool
        :param average_res: Whether or not we should average our results during validation
        :type average_res: Bool
        :return: Either 1 or 0 to relate to correct or incorrect or the probability returned from the network and
        the loss
        :rtype: list
        """

        video = sample['video']
        label = sample['label'].to(self.device).unsqueeze(0)
        filename = sample['filename']
        label = label.unsqueeze(0)
        if self.segment:
            boxes = sample[self.segment]

        losses = []
        accuracies = []
        segment_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_list = []

        frames_to_get = random.randrange(self.frame_seg, segment_length, self.frame_seg)

        start_frame = frames_to_get - self.frame_seg
        end_frame = frames_to_get
        current_frame = 0

        while (True):

            # Capture frame-by-frame
            ret, frame = video.read()
            batch = torch.empty(0)

            # Try and catch used to work out when video stream ends
            try:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if current_frame < start_frame and not average_res:
                    current_frame += 1
                    continue

                if self.segment:
                    box = boxes[current_frame]

                    if sum(box) > 0:
                        frame = frame.crop(box)
                    else:
                        boxes.pop(current_frame)
                        continue

                if train:
                    frame = self.tr_transforms(frame).to(self.device)
                else:
                    frame = self.te_transforms(frame).to(self.device)

            except Exception as e:

                if average_res:
                    total_loss = 0
                    out = 0
                    for i in range(0, len(losses)):
                        total_loss += losses[i]
                        out += accuracies[i][0]

                    output = out/len(accuracies)
                    loss = total_loss/len(losses)

                    if probs:
                        return output, loss

                    answer = check_true(output, label.cpu().numpy()[0][0])
                    return answer, loss

                if current_frame < end_frame:
                    if debug:
                        print(e)
                        print(filename)
                    return -1

            current_frame += 1
            frame_list.append(frame)

            if (average_res and current_frame % self.frame_seg == 0) or current_frame == end_frame:
                torch.stack(frame_list, out=batch)

                batch = batch.unsqueeze(0)
                if batch.is_cuda is False:
                    batch = batch.to(self.device)

                if train:
                    output = self.net(batch)
                else:
                    output = self.net(batch, dropout=False)

                frame_list.clear()
                loss = self.criterion(output, label)

                if train:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()

                if average_res:
                    output = output.detach()
                    torch.sigmoid(input=output, out=output)
                    accuracies.append(output.cpu().numpy()[0])
                    losses.append(loss.item())
                else:
                    output = output.detach()
                    torch.sigmoid(input=output, out=output)
                    if probs:
                        return output, loss.item()
                    answer = check_true(output.cpu().numpy()[0][0], label.cpu().numpy()[0][0])
                    return answer, loss.item()
                del output

def check_true(answer, label):
    """
    Check if a given answer matches the label
    :param answer: the probability output from the network
    :type answer: float
    :param label: the label integer as either a 1 or a 0
    :type label: float
    :return: 1 if classification is correct, 0 if incorrect
    :rtype: int
    """
    if answer > 0.5 and label == 1:
        return 1
    elif answer < 0.5 and label == 0:
        return 1
    else:
        return 0
