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
        muti = 1
        self.conv1 = nn.Conv2d(3, 16*muti, kernel_size=(5,5), stride=(2,2), padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        self.conv2 = nn.Conv2d(16*muti, 64*muti, kernel_size=(5,5), stride=(2,2), padding=2)
        self.conv3 = nn.Conv2d(64*muti, 8*muti, kernel_size=(5,5), stride=(2,2), padding=2)
        self.fc_1 = nn.Linear(32*muti, 2)

        ##

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(32))
        nn.init.constant_(self.fc_1.bias, 0.0)

        #nn.init.normal_(self.fc2.weight, 0.0, 1 / sqrt(32))
        #nn.init.constant_(self.fc2.bias, 0.0)
        ##
    
    def forward(self, x):
        N, C, H, W = x.shape
        activ_func = F.relu
        #activ_func = F.elu
        #activ_func = F.softplus
        #activ_func = F.leaky_relu
        #activ_func = torch.sigmoid
        #activ_func = torch.tanh

        ## TODO: forward pass
        muti = 1
        z = activ_func(self.conv1(x))
        z = self.pool(z)
        z = activ_func(self.conv2(z))
        z = self.pool(z)
        z = activ_func(self.conv3(z))
        z = torch.reshape(z, (N, 32*muti))
        z = self.fc_1(z)
        
        ##

        return z
