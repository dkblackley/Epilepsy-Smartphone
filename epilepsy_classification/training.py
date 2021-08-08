
from tqdm import tqdm
import random
import cv2
import csv
import torch
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import epilepsy_classification.model as model
import torch.nn as nn
import utils

class Trainer:

    def __init__(self, dataset, frame_segments, transforms, net=False, segment='', optimizer=False):

        self.dataset = dataset
        self.frame_seg = frame_segments
        self.criterion = nn.BCEWithLogitsLoss()
        self.transforms = transforms
        self.segment = segment

        if not net:
            self.net = model.Classifier(frame_segments, dropout=0.35)
            self.optim = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)


    def train(self, epochs, shuffle=True, debug=False):

        indices_original = [i for i in range(0, len(self.dataset))]
        t_overall_accuracy = []
        v_overall_accuracy = []
        t_overall_loss = []
        v_overall_loss = []
        indices = indices_original.copy()


        if shuffle:
            random.shuffle(indices)

        train = indices[int(len(indices) * 0.25):]
        val = indices[:int(len(indices) * 0.25)]

        for i in range(0, epochs):

            print(f"EPOCH {i} of {epochs}")

            accuracies = []
            losses = []
            for current in tqdm(train):

                data = self.dataset[current]
                self.net.reset_states()

                if not utils.check_number_of_boxes(data[self.segment], debug=True):
                    continue

                answer, loss = self.run_through(data, True)

                accuracies.append(answer)
                losses.append(loss)

            print(losses)
            print(accuracies)
            t_overall_loss.append(sum(losses)/len(losses))
            t_overall_accuracy.append(sum(accuracies)/len(accuracies))

            accuracies = []
            losses = []
            for current in tqdm(val):

                data = self.dataset[current]
                self.net.reset_states()

                answer, loss = self.run_through(data, True)

                accuracies.append(answer)
                losses.append(loss)

            print(losses)
            print(accuracies)
            v_overall_loss.append(sum(losses) / len(losses))
            v_overall_accuracy.append(sum(accuracies) / len(accuracies))

            print('Train acc:')
            print(t_overall_accuracy)
            print('Train loss:')
            print(t_overall_loss)
            print('val acc:')
            print(v_overall_accuracy)
            print('val loss:')
            print(v_overall_loss)



    def run_through(self, sample, train, debug=False):

        video = sample['video']
        label = sample['label']
        filename = sample['filename']
        label = label.unsqueeze(0)

        if self.segment:
            boxes = sample[self.segment]

        losses = []
        accuracies = []

        segment_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_list = []
        frame_count = 0

        if train:
            self.net.train()
        else:
            self.net.eval()

        while (True):

            # Capture frame-by-frame
            ret, frame = video.read()
            batch = torch.empty(0)

            try:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if self.segment:
                    box = boxes.pop(0)

                if sum(box) > 0:
                    frame = frame.crop(box)
                else:
                    continue
                frame = self.transforms(frame)
            except:

                if not frame_list:
                    if torch.argmax(output) == torch.argmax(label):
                        return 1, loss.item()
                    else:
                        return 0, loss.item()

                torch.stack(frame_list, out=batch)
                frame_list.clear()
                losses.append(loss.item())

                if torch.argmax(output) == torch.argmax(label):
                    return 1, loss.item()
                else:
                    return 0, loss.item()

            frame_list.append(frame)

            frame_count += 1

            if frame_count % self.frame_seg == 0:
                torch.stack(frame_list, out=batch)

                batch = batch.unsqueeze(0)
                output = self.net(batch)
                frame_list.clear()
                loss = self.criterion(output, label)

                if train:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                #if frame_count > 50:
                    #self.net.reset_states()

    def test(self):
        pass