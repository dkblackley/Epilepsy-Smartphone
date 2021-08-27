"""
model.py - File responsible for holding the model
"""
import torch
import torch.nn as nn
from torch.nn import functional as TF

__author__     = ["Daniel Blackley"]
__copyright__  = "Copyright 2021, Epilepsy-Smartphone"
__credits__    = ["Daniel Blackley", "Stephen McKenna", "Emanuele Trucco"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@gmail.com"
__status__     = "Development"


class Classifier(nn.Module):
    """
    Classifier that inherits from nn.Module, very shallow architecture, holding only a single convolutional layer and
    a single LSTM layer
    """

    def __init__(self, frame_length, dropout=0.5, device='cpu'):
        """
        Sets default values for model
        :param frame_length: number of frames for the LSTM layer
        :type frame_length: int
        :param dropout: Drop rate for the network
        :type dropout: float
        :param device: device to place LSTM hidden layers on
        :type device: str
        """
        super(Classifier, self).__init__()
        self.drop_rate = dropout
        self.lstm_size = 64
        self.conv_output = 64
        self.device = device

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

                torch.zeros(num_layers, batch_size, self.lstm_size).requires_grad_().to(self.device),
                torch.zeros(num_layers, batch_size, self.lstm_size).requires_grad_().to(self.device)
            ]
        else:
            self.hidden1 = [
                torch.zeros(num_layers, batch_size, self.lstm_size).to(self.device),
                torch.zeros(num_layers, batch_size, self.lstm_size).to(self.device)
            ]

    def forward(self, input, detach=True, dropout=True):
        """
        forward pass through the network
        :param input: the input vector, expected to be of size (batch size, frame length, image size 1, image size 2)
        :type input: Pytorch Tensor
        :param detach: Whether or not to detach and use the hidden layers again for the next input.
        Usually only useful for validation/testing
        :type detach: Bool
        :param dropout: Whether dropout should be applied
        :type dropout: Bool
        :return: The single value that is output from the network. < 0.5 implies spasms, > 0.5 implies mimics
        :rtype: Pytorch Tensor
        """

        batch_size, timesteps, C, H, W = input.size() # Use these for changing Tensor size

        output = input.view(batch_size * timesteps, C, H, W) # instead of 1 batch of 60 frames make 60 batches

        output = self.pool(self.activation(self.bn(self.conv(output))))
        output = output.view(timesteps, batch_size, -1) # Change back to 1 batch of 60 frames
        output, hidden1 = self.rnn1(output, self.hidden1)

        if dropout:
            output = TF.dropout(output, self.drop_rate)

        output = output.reshape(output.shape[1], -1)

        if detach:
            self.hidden1[0] = hidden1[0].detach().to(self.device)
            self.hidden1[1] = hidden1[1].detach().to(self.device)

        if dropout:
            output = TF.dropout(output, self.drop_rate)

        output = self.output_layer(output)

        return output
