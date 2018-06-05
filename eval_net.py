#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:42:19 2018

@author: vl-tshzzz
"""

import torch
from torch.autograd import Variable
import torch.nn as nn


from tqdm import tqdm

import torchvision.transforms as transforms
import cv2
import numpy as np
from network import YOLO,TinyYOLO
import torch.nn.functional as F
import voc_datasets

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#classes = ["person"]


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    #print(boxes)
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes

def decoder_(pred):
    bb_num = 3
    cls_num = len(classes)
    pred = pred.view(-1,7,7,bb_num*5+cls_num)
    prob = pred[:,:,:,4:bb_num*5:5]
    max_prob,max_prob_index = prob.max(3)
    cell_size = 1./7
    
    boxes=[]

    for k in range(len(pred)):
        for i in range(7):
            for j in range(7):  
                cls_prob,cls = pred[k,i,j,5*bb_num:].max(0)
                if max_prob[k,i,j].data.numpy()* cls_prob.data  > 0.2:
                    max_prob_index_np = max_prob_index[k,i,j].data.numpy()

                    bbox = pred[k , i , j , max_prob_index_np*5 : max_prob_index_np*5 + 5+1].contiguous().data   
                    
                    bbox[5] = int(cls.data.numpy())
                    
                    box_xy = torch.FloatTensor(bbox.size())
                    box_xy[:2] = bbox[:2] - 0.5*pow(bbox[2:4],2)
                    box_xy[2:4] = bbox[:2] + 0.5*pow(bbox[2:4],2)
                    box_xy[4] = max_prob[k,i,j].data * cls_prob.data
                    box_xy[5] = bbox[5]
                    boxes.append(box_xy.view(1,6).numpy()[0].tolist())
    boxes = nms(boxes, 0.5)
    return boxes

def predict_gpu(model,img):

    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    bbox = decoder_(pred)
    bbox = np.array(bbox)
    return bbox
 

def test_net(model,loader,iou_thresh=0.5):
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    eps = 1e-5
    ap = []
    for images,label in loader:
        label = label.view(-1,6)
        images = images.cuda()
        pred_boxes = predict_gpu(model,images)
        correct_this = 0.0
        precision_this = 0.0  
        
        num_gts = len(label)
        total = total + num_gts
        proposals += len(pred_boxes)
            
        for i in range(num_gts):
                
            box_gt = [label[i][0], label[i][1], label[i][2], label[i][3], label[i][4], label[i][5]]
                
            best_iou = 0
            best_j = -1
            for j in range(len(pred_boxes)):
                iou = bbox_iou(box_gt, pred_boxes[j], x1y1x2y2=True)
                if iou > best_iou:
                    best_j = j
                    best_iou = iou
            
            if best_iou > iou_thresh and int(pred_boxes[best_j][5]) == int(box_gt[5]):
                correct = correct+1
                correct_this += 1.0
                    
        precision_this = correct_this/(len(pred_boxes)+eps)
        recall_this = correct_this/(num_gts+eps)
        ap.append(min(precision_this,recall_this))
    '''    
    ap = np.array(ap) 
    area = 0.0
    for i in range(len(ap) - 1):
        area += (ap[i] + ap[i+1])*1./2.
    '''
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    
    #print("ap: %f "%(area/len(train_iterator) ))
    print("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
    
if __name__ == '__main__':

    model = YOLO(20,3,conv_model = True)
    model.load_state_dict(torch.load('./models/conv_model_.pkl480'))
    model.cuda()
    model.eval()
    batch_size = 1
    iou_thresh = 0.5
    


    test_loader =  voc_datasets.get_test_loader('./','train_list/test_voc.txt',448,batch_size,8)
    train_iterator = tqdm(test_loader)
    
    test_net(model,train_iterator,iou_thresh=0.5)
    

    
    

    
    