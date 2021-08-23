
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
import torch

class Trainer:

    def __init__(self, dataset, frame_segments, tr_transforms, te_transforms, net=False, segment='', save_diir="models/",
                 optimizer=False, threshold=0.75, early_stop=True):

        self.dataset = dataset
        self.frame_seg = frame_segments
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion2 = nn.BCEWithLogitsLoss()
        self.weight = torch.tensor([1.0, 1.0])
        self.tr_transforms = tr_transforms
        self.te_transforms = te_transforms
        self.segment = segment
        self.save_dir = save_diir
        self.threshold = threshold
        self.early_stop = early_stop

        if not net:
            self.net = model.Classifier(frame_segments, dropout=0.35)
            self.optim = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)
        else:
            self.net = net
            self.optim = optimizer

    def set_save_dir(self, new_dir):
        self.save_dir = new_dir

    def set_weights(self, indices):

        x1 = 0
        x2 = 0
        for i in indices:
            data = self.dataset[i]
            label = data['label']
            video = data['video']
            segment_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if np.argmax(label) == 0:
                x1 += segment_length
            else:
                x2 += segment_length

        x1 = 1/x1
        x2 = 1/x2

        if x1 > x2:
            #x1 *= 100
            x2 = x2 / x1
            x1 = x1/x1
        else:
            #x2 *= 2
            x1 = x1/x2
            x2 = x2/x2

        self.weight = torch.tensor([x1, x2])


    def LOSO(self, epochs, shuffle=True, debug=False):

        indices = [i for i in range(0, len(self.dataset))]

        i = 0

        if debug:
            print(f"{len(indices)} number of videos before purge")

        while i < len(indices):
            data = self.dataset[indices[i]]

            #TODO remove videos with less than 80% detection
            if self.segment and not utils.num_boxes_greater_than_ratio(data[self.segment], ratio=self.threshold, debug=debug):
                if debug:
                    print(f"{data['filename']} Has been removed\n")
                indices.pop(i)
            elif self.segment:
                if len(data[self.segment]) < self.frame_seg:
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

        for i in range(0, len(indices)):
            self.save_dir = f"models/LOSO_model_{self.segment}_{i + 1}/"
            if debug:
                print(f"Working on model LOSO_model_{self.segment}_{i + 1}")
            self.net = model.Classifier(self.frame_seg, dropout=0.35)
            self.optim = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)
            indices_copy = indices.copy()
            self.train(epochs, train=indices_copy[:i] + indices_copy[i+1:], val=[indices[i]])

            results.append(self.test([indices[i]])[0])

        utils.write_to_csv(f"models/LOSO_{self.segment}_RESULTS.csv", results)



    def train(self, epochs, shuffle=True, train=None, val=None, split=0.25, debug=False):

        t_overall_accuracy = []
        v_overall_accuracy = []
        t_overall_loss = []
        v_overall_loss = []

        if train is None:
            indices_original = [i for i in range(0, len(self.dataset))]
            train = indices_original[int(len(indices_original) * split):].copy()
            val = indices_original[:int(len(indices_original) * split)].copy()

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

                """if self.segment:
                    if not utils.num_boxes_greater_than_ratio(data[self.segment], debug=True) or len(data[self.segment]) < self.frame_seg:
                        if debug:
                            print(f"Skipping video {data['filename']} due to unsatisfactory bounding boxes\n")
                        continue"""

                answer, loss = self.run_through(data, True)

                accuracies.append(answer)
                losses.append(loss)

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
                    answer, loss = self.run_through(data, False)

                accuracies.append(answer)
                losses.append(loss)

            print(losses)
            print(accuracies)
            v_overall_loss.append(sum(losses) / len(losses))
            v_overall_accuracy.append(sum(accuracies) / len(accuracies))

            if (sum(losses)/len(losses)) <= min(v_overall_loss) and self.early_stop:
                utils.save_model(self.net, self.optim, self.save_dir + "best_loss/")
                utils.save_results(self.save_dir + "best_loss/", [v_overall_loss], [v_overall_accuracy],
                                   loss_filename="val_losses", accuracry_filname="val_accuracy")
                utils.save_results(self.save_dir + "best_loss/", [t_overall_loss], [t_overall_accuracy])


            utils.save_model(self.net, self.optim, self.save_dir + "model/")
            utils.save_results(self.save_dir + "model/", [v_overall_loss], [v_overall_accuracy],
                               loss_filename="val_losses", accuracry_filname="val_accuracy", debug=debug)
            utils.save_results(self.save_dir + "model/", [t_overall_loss], [t_overall_accuracy], debug=debug)

            print('\nTrain acc:')
            print(t_overall_accuracy)
            print('Train loss:')
            print(t_overall_loss)
            print('\nval acc:')
            print(v_overall_accuracy)
            print('val loss:')
            print(v_overall_loss)

    def test(self, indices):

        self.net.eval()
        results = []

        for index in indices:
            data = self.dataset[index]
            self.net.reset_states(requires_grad=False)

            with torch.no_grad():
                answers, _ = self.run_through(data, False, probs=True)

            answers = answers.tolist()

            for answer in answers:
                result = [data['filename'], answer[0], answer[1]]
                results.append(result)

        return results



    def run_through(self, sample, train, debug=False, probs=False):

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
                        #if filename == "spasm6.mp4":
                        #    frame.show()
                    else:
                        continue
                if train:
                    frame = self.tr_transforms(frame)
                else:
                    frame = self.te_transforms(frame)
            except:

                if not frame_list:

                    if probs:
                        output = output.detach()
                        torch.sigmoid(input=output, out=output)
                        return output, loss.item()

                    if torch.argmax(output) == torch.argmax(label):
                        return 1, loss.item()
                    else:
                        return 0, loss.item()

                torch.stack(frame_list, out=batch)
                frame_list.clear()
                losses.append(loss.item())

                if probs:
                    output = output.detach()
                    torch.sigmoid(input=output, out=output)
                    return output, loss.item()

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
                loss_test = self.criterion2(output, label)
                loss = self.criterion(output, label)
                loss = loss * self.weight
                loss = loss.mean()

                if train:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                #if frame_count > 50:
                    #self.net.reset_states()