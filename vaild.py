#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:05:18 2018

@author: vl-tshzzz
"""

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
                if max_prob[k,i,j].data.numpy()* cls_prob.data  > 0.1:
                    max_prob_index_np = max_prob_index[k,i,j].data.numpy()

                    bbox = pred[k , i , j , max_prob_index_np*5 : max_prob_index_np*5 + 5+len(classes)].contiguous().data   
                    
                    #bbox[5] = int(cls.data.numpy())
                    
                    box_xy = torch.FloatTensor(bbox.size())
                    box_xy[:2] = bbox[:2] - 0.5*pow(bbox[2:4],2)
                    box_xy[2:4] = bbox[:2] + 0.5*pow(bbox[2:4],2)
                    box_xy[4] = max_prob[k,i,j].data #* cls_prob.data
                    box_xy[5:] = pred[k,i,j,5*bb_num:].data
                    boxes.append(box_xy.view(1,5+len(classes)).numpy()[0].tolist())
    boxes = nms(boxes, 0.5)
    return boxes

def predict_gpu(img):

    pred = model(img) #1x7x7x30
    #pred = pred.view(-1,7,7,11)
    pred = pred.cpu()
    bbox = decoder_(pred)
    bbox = np.array(bbox)
    return bbox
 


if __name__ == '__main__':
    import os
    prefix = 'results'
    outfile = "voc"
    
    test_list = "VOC2007_train.txt"
    
    class_num = len(classes)#20
    model = YOLO(class_num,3,conv_model = False)
    model.load_state_dict(torch.load('./models/fc_models/model_.pkl116'))
    model.cuda()
    model.eval()
    batch_size = 1
    iou_thresh = 0.5
    eps = 1e-5

    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    
    test_loader =  voc_datasets.get_test_loader('./','train_list/'+test_list,448,batch_size,8)
    train_iterator = tqdm(test_loader)
    
    list_file = 'train_list/'+test_list
    lines = []
    with open(list_file) as f:
        lines = f.readlines()
    
    fps = [0]*class_num
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(class_num):
        buf = '%s/%s%s.txt' % (prefix, outfile, classes[i])
        fps[i] = open(buf, 'w')
        

    for lineId,(images,label) in enumerate(train_iterator):
        label = label.view(-1,6)
        images = images.cuda()
        pred_boxes = predict_gpu(images)
        
        fileId = os.path.basename(lines[lineId]).split('.')[0]
        
        img = cv2.imread(lines[lineId].split(" ")[0])
        width, height = img.shape[1],img.shape[0] #get_image_size(valid_files[lineId])

        for j in range(len(pred_boxes)):

            x1 = int(pred_boxes[j,0] * width)
            y1 = int(pred_boxes[j,1] * height)
            x2 = int(pred_boxes[j,2] * width)
            y2 = int(pred_boxes[j,3] * height)
            
            det_conf = pred_boxes[j,4]
            
            for k in range((len(pred_boxes[0])-5)):
                cls_conf = pred_boxes[j,5+k]
                cls_id = k
                prob = float(det_conf * cls_conf)
                fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))
                
    for i in range(class_num):
        fps[i].close()