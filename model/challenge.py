"""
EECS 445 - Introduction to Machine Learning
Winter 2021 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc_1 = nn.Linear(2048, 2)
        ##

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(32))
        nn.init.constant_(self.fc_1.bias, 0.0)
        ##
    
    def forward(self, x):
        N, C, H, W = x.shape
        #activ_func = F.relu
        activ_func = F.softplus
        ## TODO: forward pass
        z = activ_func(self.conv1(x))
        z = self.pool(z)
        z = activ_func(self.conv2(z))
        z = self.pool(z)
        z = activ_func(self.conv3(z))
        #z = self.pool(z)
        z = activ_func(self.conv4(z))
        z = torch.reshape(z, (N, 2048))
        z = self.fc_1(z)

        return z