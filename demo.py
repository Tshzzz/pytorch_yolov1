#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

import torchvision.transforms as transforms
import cv2
from network import YOLO

from box_utils import yolo_box_decoder

import config

def predict_gpu(model,image_name,root_path=''):

    image = cv2.imread(root_path+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h,w,_ = image.shape
    img = cv2.resize(image,(448,448)) 
    


    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [1, 1, 1])])
    img = transform(img)
    img = Variable(img[None,:,:,:])
    img = img.cuda()

    pred= model(img)

    boxes,classes = yolo_box_decoder(pred,conf_thresh=0.2,nms_thresh=0.5)

    return boxes,classes
 
def plot_boxes_cv2(img, boxes,classes, savename=None, class_names=None, color=None):

    width = img.shape[1]
    height = img.shape[0]
    
    for j in range(len(boxes)):
        
        x1 = boxes[j,0]
        y1 = boxes[j,1]
        x2 = x1 + boxes[j,2]
        y2 = y1 + boxes[j,3]
        x1,x2 = int(x1*width),int(x2*width)
        y1,y2 = int(y1*height),int(y2*height)
        
        cls_id = int(classes[j,0])
        prob = float(classes[j,1])
        
        img = cv2.putText(img, class_names[cls_id], (x1,y1),  cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), thickness=1)
        img = cv2.putText(img, str(round(prob,3)), (x1,y1+20),  cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), thickness=1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
            
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img,len(boxes)


if __name__ == '__main__':

    import os

    model = YOLO(config.cls_num,config.bbox_num,config.box_scale,conv_model = config.use_conv)
    model.eval()
    model.load_state_dict(torch.load('./model.pkl'))
    model.cuda()
    
    
    test_path = list(map(lambda x: os.path.join('./test', x), os.listdir('./test')))
    for image_name in test_path:
        save_name = image_name.split('/')[-1]
        image = cv2.imread(image_name)
        
        
        boxes,classes = predict_gpu(model,image_name)
        image,num = plot_boxes_cv2(image, boxes, classes,savename='samples/'+save_name, class_names=config.classes, color=None)
        cv2.namedWindow('result')
        cv2.imshow('result',image)
        cv2.waitKey()

    
    
    