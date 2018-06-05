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

from eval_net import test_net

epochs_start = 7
epochs_end = 9
batch_size = 32
cls_num = 20
max_update_batch = 1


from network import YOLO


net = YOLO(cls_num,3,conv_model = False)
net.cuda()
net.train()
if epochs_start > 0:
    net.load_state_dict(torch.load('./models/fc_models/model_.pkl{0}'.format(epochs_start)))
print("load model successfully")

train_loader =  voc_datasets.get_loader('./','train_list/train_voc.txt',True,448,batch_size,8)
test_loader =  voc_datasets.get_test_loader('./','train_list/eval_train_voc.txt',448,1,3)

start_lr = 3e-3#0.0005 0.00005
#0.005
optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=0.9, weight_decay=5e-4)

scheduler = MultiStepLR(optimizer, milestones=[1,2,3],gamma=1.5)

loss_detect = yolov1_loss(7,3,5,0.5,cls_num)
logger = Logger('./logs')

step = epochs_start * len(train_loader)

for epoch in range(epochs_start,epochs_end):
    epoch_loss = 0
    train_iterator = tqdm(train_loader)
    mulit_batch_ = 0
    
    for train_batch,(images,label) in enumerate(train_iterator):
        label = label.cuda()
        label = Variable(label)
        
        images = images.cuda()
        images = Variable(images)
        
        
        pred = net(images).type(torch.cuda.DoubleTensor)

        #if mulit_batch_ == 0:
        loss_xx,loss_info = loss_detect(pred,label)
        #mulit_batch_ = loss_xx
        #else:
        #    loss_xx,loss_info = loss_detect(pred,label)
        #    mulit_batch_ += loss_xx
            
        epoch_loss += loss_xx
        
        status = '[{0}] lr = {1} batch_loss = {2} epoch_loss = {3} '.format(
                epoch + 1,scheduler.get_lr()[0], loss_xx.data, epoch_loss.data/(train_batch+1))
        
        train_iterator.set_description(status)

        for tag, value in loss_info.items():
            logger.scalar_summary(tag, value, step)
        loss_xx.backward()      
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        
    if epoch % 5 == 0 and epoch > 10:
        print("Evaluate~~~~~   ")
        net.eval()
        test_net(net,test_loader,iou_thresh=0.5)
        net.train()
    if epoch % 5 == 0:    
        torch.save(net.state_dict(),"./models/fc_models/model_.pkl"+repr(epoch+1))
    scheduler.step()
    
    
    
torch.save(net.state_dict(),"./models/fc_models/model_.pkl"+repr(epoch+1))














