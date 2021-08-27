"""
File responsible for holding the model, uses efficientnet
"""
import torch
import torch.nn as nn
from torch.nn import functional as TF


class Classifier(nn.Module):
    """
    Classifier that inherits from nn.Module, very shallow architecture, holding only a single convolutional layer and
    a single LSTM layer
    """

    def __init__(self, frame_length, dropout=0.5):
        """
        Sets default values for model
        :param frame_length: number of frames for the LSTM layer
        :type frame_length: int
        :param dropout: Drop rate for the network
        :type dropout: int
        """
        super(Classifier, self).__init__()
        self.drop_rate = dropout
        self.lstm_size = 64
        self.conv_output = 64

        self.conv = nn.Conv2d(3, self.conv_output, kernel_size=(3, 3), padding=(1,1))
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.rnn = nn.LSTM(self.conv_output, self.lstm_size, 1, batch_first=False, dropout=dropout)
        self.output_layer = nn.Linear(self.lstm_size * frame_length, 1)
        self.activation = torch.nn.LeakyReLU()

    def reset_states(self, num_layers=1, batch_size=1, requires_grad=True):
        """
        Used to reset the hidden states inside the LSTM Layer. Means that you can feed an entire video in, even if you
        don't have memory to hold each image. You can't train this way however, as you still need to hold the gradients
        :param num_layers: number of layers in the LSTM
        :type num_layers: int
        :param batch_size: batch_size
        :type batch_size: int
        :param requires_grad: whether or not we're training the network
        :type requires_grad: Bool
        """

        if requires_grad:
            self.hidden1 = [
                torch.zeros(num_layers, batch_size, self.lstm_size).requires_grad_(),
                torch.zeros(num_layers, batch_size, self.lstm_size).requires_grad_()
            ]
        else:
            self.hidden1 = [
                torch.zeros(num_layers, batch_size, self.lstm_size),
                torch.zeros(num_layers, batch_size, self.lstm_size)
            ]

    def forward(self, input, detach=True, dropout=True):
        """
        forward pass through the network
        :param input:
        :type input:
        :param detach:
        :type detach:
        :param dropout:
        :type dropout:
        :return:
        :rtype:
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
        #output = self.pool2(self.activation(self.bn2(self.conv2(output))))
        #output = self.pool3(self.activation(self.bn3(self.conv3(output))))

        """output = output.view(-1) # use for flatten
        output = output.unsqueeze(0) # use for flatten
        output = self.activation(self.fc1(output))
        TF.dropout(output, self.drop_rate)"""

        """output = output.view(-1, self.shape)
        output = output.unsqueeze(0)
        output = self.activation(self.fc1(output))
        if dropout:
            output = TF.dropout(output, self.drop_rate)"""

        output = output.view(timesteps, batch_size, -1)
        output, hidden1 = self.rnn1(output, self.hidden1)
        if dropout:
            output = TF.dropout(output, self.drop_rate)
        output = output.reshape(output.shape[1], -1)

        if detach:
            self.hidden1[0] = hidden1[0].detach()
            self.hidden1[1] = hidden1[1].detach()


        """output = self.activation(self.fc2(output))
        if dropout:
            output = TF.dropout(output, self.drop_rate)"""

        output = self.output_layer(output)

        return output
