# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import sys


class DeepCNN(nn.Module):
    """CNN for predicting CCBSS"""
    def __init__(self, in_channels, motiflen=20):
        super(DeepCNN, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # classifier head
        c_in = 512 # 384
        self.bn = nn.BatchNorm1d(c_in)
        self.linear1 = nn.Linear(c_in, 32)
        self.drop = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(32, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        # classifier
        out2 = skip4.view(b, -1)
        out2 = self.bn(out2)
        out2 = self.linear1(out2)
        out2 = self.relu(out2)
        out2 = self.drop(out2)
        out2 = self.linear2(out2)
        out_class = self.sigmoid(out2)

        return out_class
