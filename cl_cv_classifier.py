# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 16:32:32 2021

@author: daliana91
"""
import torch.nn as nn
import torch
import tensorflow as tf

     
def conv_block(in_channels, out_channels, pool=False, pool_no=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_channel =14
        self.conv1 = conv_block(in_channel, 32)
        self.conv2 = conv_block(32, 64, pool=True)        
        self.conv3 = conv_block(64, 128, pool=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 4)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out