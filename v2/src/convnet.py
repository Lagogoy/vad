#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import math

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(8,4), stride=2),     # Here remains ??
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=4, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(192, 1024),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(1024, 1024), 
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(1024,2)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
#        print('CNN to Full Linear Connect:{}'.format(x.size()))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
