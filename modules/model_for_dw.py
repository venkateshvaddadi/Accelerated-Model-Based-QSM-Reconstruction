#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:12:19 2021

@author: venkatesh
"""
import torch.nn.functional as F
import os
import copy
import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets


#%%
class Dw(nn.Module):
    def __init__(self):
        super(Dw,self).__init__()

        self.lambda_val = torch.nn.Parameter(torch.rand(1), requires_grad = True)
        self.conv1=nn.Sequential(nn.Conv3d(1,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv4=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv5=nn.Sequential(nn.Conv3d(64,1,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(1))

        

    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.conv5(x4)    
        out = x + x5
        return out
    
    
#%%
"""
m = nn.Conv3d(in_channels=1, 
          out_channels=33, 
          kernel_size=3,
          stride=1,
          padding=1
          )

m = nn.Conv3d(1,33,3,1,1)
input = torch.randn(1,1,64, 64, 64)
output = m(input)

#%%

input = torch.randn(1,1,64, 64, 64)

conv1=nn.Sequential(nn.Conv3d(1,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())

net=Dw()
#print('output network shape',net(input).shape)


#%%

# testing for the batchsize greater than one
# batch size is 8

input = torch.randn(8,1,64, 64, 64)
model=Dw().cuda()
input=input.cuda()
"""

#%%
# testing for the input which is using fully convolution layer or not?
# we are checking ..............






# input = torch.randn(1,1,176, 176, 160)
# model=Dw().cuda()

# input=input.cuda()
# output=model(input).cpu()
# #%%
# print(model.lambda_val.grad)





