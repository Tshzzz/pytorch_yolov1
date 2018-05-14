#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:12:23 2018

@author: vl-tshzzz
"""

import torch
from logger import Logger


import voc_datasets



from torch.autograd import Variable
import torch.optim as optim

from tqdm import tqdm

from loss import yolov1_loss
from torch.optim.lr_scheduler import MultiStepLR

epochs_start = 0
epochs_end = 256
batch_size = 30
cls_num = 1


from network import YOLO


net = YOLO(cls_num,3)
net = net.cuda()

if epochs_start > 0:
    net.load_state_dict(torch.load('./models/model_.pkl{0}'.format(epochs_start)))
print("load model successfully")
test_loader =  voc_datasets.get_loader('./','train_list/person.txt',True,448,batch_size,8)
start_lr = 1e-4
milestones = [80,150]
optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=0.9, weight_decay=5e-3)
scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones])
loss_detect = yolov1_loss(7,3,5,0.5,cls_num)
logger = Logger('./logs')

step = epochs_start * len(test_loader)



for epoch in range(epochs_start,epochs_end):
    epoch_loss = 0
    train_iterator = tqdm(test_loader)
    count = 1.
    
    for images,label in train_iterator:
        label = label.cuda()
        label = Variable(label)
        
        images = images.cuda()
        images = Variable(images)
        
        optimizer.zero_grad()
        pred = net(images).type(torch.cuda.DoubleTensor)

        loss_xx,loss_info = loss_detect(pred,label)

        epoch_loss += loss_xx.data
        count += 1
        
        status = '[{0}] lr = {1} batch_loss = {2} epoch_loss = {3} '.format(
                epoch + 1,scheduler.get_lr()[0], loss_xx.data, epoch_loss/count)
        
        train_iterator.set_description(status)

        for tag, value in loss_info.items():
            logger.scalar_summary(tag, value, step)
            
        loss_xx.backward()  
        optimizer.step()
        
        step += 1
    if epoch % 1 == 0:    
        torch.save(net.state_dict(),"./models/model_.pkl"+repr(epoch+1))
    scheduler.step()
