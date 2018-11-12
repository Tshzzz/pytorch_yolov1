#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:12:23 2018

@author: vl-tshzzz
"""

import torch
import voc_datasets
import torch.optim as optim
from tqdm import tqdm
from loss import yolov1_loss
from torch.optim.lr_scheduler import MultiStepLR
from network import YOLO
from vaild import eval_mAp
import config
import numpy as np
from tensorboardX import SummaryWriter


epochs_start = config.epochs_start
epochs_end = config.epochs_end
batch_size = config.batch_size
cls_num = config.cls_num
bbox_num = config.bbox_num
start_lr = config.lr
train_image_list = config.train_image_list
eval_image_list= config.eval_image_list
model_path = config.model_path
model_save_iter = config.model_save_iter
l_coord = config.l_coord
l_noobj = config.l_noobj
scale_size = config.box_scale
conv_model = config.use_conv

pretrained = config.pretrain_path


if epochs_start == 0:
    net = YOLO(cls_num,bbox_num,scale_size,conv_model = conv_model,pretrained = pretrained)
else:
    net = YOLO(cls_num,bbox_num,scale_size,conv_model = conv_model)
    
net.cuda()
net.train()

if epochs_start > 0:
    net.load_state_dict(torch.load(model_path+'model_.pkl'))
print("load model successfully")

train_loader = voc_datasets.get_loader(train_image_list,448,batch_size,True,8)

#test_loader = voc_datasets.get_loader(eval_image_list,448,1,False,8)

optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[30,105],gamma=0.1)


loss_detect = yolov1_loss(bbox_num,l_coord,l_noobj,cls_num)
logger = SummaryWriter()
step = epochs_start * len(train_loader)

best_score = 0 
eval_score = 0


for epoch in range(epochs_start,epochs_end):
    epoch_loss = 0
    train_iterator = tqdm(train_loader)
    mulit_batch_ = 0
    
        
    for train_batch,(images,label_cls,label_response,label_bboxes,fname) in enumerate(train_iterator):
        
        label_cls = label_cls.cuda().float()
        label_response = label_response.cuda().float()
        label_bboxes = label_bboxes.cuda().float()
        
        images = images.cuda()
        
        pred_cls,pred_response,pred_bboxes = net(images)

        loss_xx,loss_info = loss_detect(pred_cls,pred_response,pred_bboxes,label_cls,label_response,label_bboxes)

        assert not np.isnan(loss_xx.data.cpu().numpy())

        epoch_loss += loss_xx
        
        status = '[{0}] lr = {1} batch_loss = {2} epoch_loss = {3} '.format(
                epoch + 1,scheduler.get_lr()[0], loss_xx.data, epoch_loss.data/(train_batch+1))
        
        train_iterator.set_description(status)

        for tag, value in loss_info.items():
            logger.add_scalar(tag, value, step)
            
        loss_xx.backward()  


        optimizer.step()
        optimizer.zero_grad()
        step += 1

    
    if epoch % 1 == 0 :
        print("Evaluate~~~~~   ")
        net.eval()
        result = eval_mAp(net,'results','',eval_image_list)
        eval_score = np.mean(list(result.values()))
        
        
        net.train()
        
    if best_score < eval_score:  
        best_score = eval_score 
        torch.save(net.state_dict(),model_path+"model_.pkl")
        
    #print("precision: %f, recall: %f, fscore: %f, best_f1: %f" % (precision, recall, eval_score,best_score))
    print("mean ap : {} , best ap: {}".format(eval_score,best_score))
    scheduler.step()
    
    
print(best_score)    
    
torch.save(net.state_dict(),model_path+"model_.pkl"+repr(epoch+1))




