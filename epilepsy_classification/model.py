"""
File responsible for holding the model, uses efficientnet
"""
import torch
import torch.nn as nn
from torch.nn import functional as TF
from efficientnet_pytorch import EfficientNet
#import convlstm


class Classifier(nn.Module):
    """
    Class that holds and runs the efficientnet CNN
    """

    def __init__(self, frame_length, dropout=0.5):
        """
        Initialises network parameters
        :param image_size: Input image size, used to calculate output of efficient net layer
        :param output_size: number of classes to classify
        :param class_weights: the weights assigned to each class, used for BBB
        :param device: cpu or gpu, used for BBB
        :param hidden_size: size of first hidden layer
        :param dropout: Drop rate
        :param BBB: Whether or not to make layers Bayesian
        """
        super(Classifier, self).__init__()
        self.drop_rate = dropout
        self.lstm_size = 64

        #self.convlstm = convlstm.ConvLSTM(input_dim=3, hidden_dim=[128, 64], kernel_size=(3,3), num_layers=2, batch_first=True)

        #self.conv1 = nn.Conv2d(3, 128, 5)

        #padding = self.calc_padding(1, 128, 224, 5)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(6, 6), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        #self.pool1 = nn.MaxPool2d((5, 5))
        self.pool1 = nn.AdaptiveAvgPool2d((2,2))
        #self.embed = EfficientNet.from_pretrained("efficientnet-b0")

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        #self.pool2 = nn.MaxPool2d((5, 5))
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        self.conv3 = nn.Conv2d(64, 256, kernel_size=(3,3), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(512)
        #self.pool3 = nn.MaxPool2d(1)

        """with torch.no_grad():
            temp_input = torch.zeros(1, 3, 224, 224)
            temp_input = self.pool1(self.conv1((temp_input)))
            temp_input = self.pool2(self.conv2((temp_input)))
            self.shape = temp_input[0].shape[0]*temp_input[0].shape[1]*temp_input[0].shape[2]"""

        #self.fc1 = nn.Linear(self.shape, 128)

        self.rnn1 = nn.LSTM(64, self.lstm_size, 1, batch_first=False, dropout=dropout)
        #self.rnn2 = nn.LSTM(128, 64)

        #self.fc1 = nn.Linear(self.lstm_size * frame_length, 256)
        self.fc2 = nn.Linear(self.lstm_size * frame_length, 64)
        #self.fc2 = nn.Linear(1000, 500) #hidden size * number of frames
        self.output_layer = nn.Linear(64, 2)

        #self.output_size = output_size
        self.activation = torch.nn.ReLU()

        #print(f"Hidden layer size: {hidden_size}")

        """with torch.no_grad():

            temp_input = torch.zeros(1, 3, image_size, image_size)
            encoder_size = self.model.extract_features(temp_input).shape[1]"""

    def calc_padding(self, stride, W_out, W_in, filter_size):

        pad = (stride * (W_out - 1)) - (W_in - filter_size)

        return pad/2

    def freeze_efficientNet(self, requires_grad):
        self.embed.requires_grad = requires_grad

    def reset_states(self, num_layers=1, batch_size=1, requires_grad=True):
        #x.size(0)
        hidden_size = self.lstm_size
        if requires_grad:
            self.hidden1 = [
                torch.zeros(num_layers, batch_size, hidden_size).requires_grad_(),
                torch.zeros(num_layers, batch_size, hidden_size).requires_grad_()
            ]
        else:
            self.hidden1 = [
                torch.zeros(num_layers, batch_size, hidden_size),
                torch.zeros(num_layers, batch_size, hidden_size)
            ]

    def forward(self, input, detach=True):
        """
        Extracts efficient Net output then passes it through our other layers
        :param input: input Image batch
        :return: the output of our network
        """

        batch_size, timesteps, C, H, W = input.size()

        # batch_size = input.size(0)
        #output = self.pool1(self.activation(self.bn1(self.conv1(output))))

        output = input.view(batch_size * timesteps, C, H, W)

        """output = self.conv1(output)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.pool1(output)"""
        output = self.pool1(self.activation(self.bn1(self.conv1(output))))
        output = self.pool2(self.activation(self.bn2(self.conv2(output))))
        #output = self.pool3(self.activation(self.bn3(self.conv3(output))))

        """output = output.view(-1) # use for flatten
        output = output.unsqueeze(0) # use for flatten
        output = self.activation(self.fc1(output))
        TF.dropout(output, self.drop_rate)"""

        """output = output.view(-1, self.shape)
        output = self.activation(self.fc1(output))
        output = TF.dropout(output, self.drop_rate)"""

        output = output.view(timesteps, batch_size, -1)
        output, hidden1 = self.rnn1(output, self.hidden1)
        output = TF.dropout(output, self.drop_rate)
        output = output.reshape(output.shape[1], -1)

        if detach:
            self.hidden1[0] = hidden1[0].detach()
            self.hidden1[1] = hidden1[1].detach()


        output = self.activation(self.fc2(output))
        output = TF.dropout(output, self.drop_rate)

        output = self.output_layer(output)

        return output
