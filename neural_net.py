#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:13:19 2022

@author: anakinsun
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F



x = torch.linspace(-1,1,100)
x = x.reshape([100,-1])
y = torch.exp(- x * x)
y_train = y + torch.randn([100,1]) * 0.03

print(x.shape,y.shape)

plt.scatter(x,y_train,c = "r",alpha = 0.8)
plt.plot(x,y)
plt.title("Data With Noise")
plt.legend(["train_data","real_data"])
plt.show()

class FFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,32)
        self.fc4 = nn.Linear(32,1)
        
    def forward(self,x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x * 2

net = FFNet()
EPOCH = 500
lr = 0.003
plt.ion()
optimizer = torch.optim.Adam(net.parameters(),lr)
loss_history = []
for epoch in range(EPOCH):
    y_hat = net(x)
    diff = y_hat - y_train
    loss = torch.sum(torch.abs(diff))
    loss_history.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    plt.cla()
    plt.plot(x,y)
    plt.scatter(x,y_train)
    plt.plot(x,y_hat.detach().numpy())
    plt.pause(0.01)
    if (epoch % 20 == 0 or epoch == EPOCH -1):
        print("Progress: [{} %] Loss : [ {} ]".format(100*(epoch+1)/EPOCH,loss.detach().numpy()))
    
plt.ioff()
plt.cla()
plt.plot(loss_history)
plt.title("Convergence History")
plt.legend(["Epoch Loss"])
    
