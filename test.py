#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

import torchvision.transforms as transforms
import cv2
import numpy as np
from network import YOLO



'''
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
'''
classes = ["person"]

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
                if max_prob[k,i,j].data.numpy() * cls_prob.data.numpy() > 0.5:
                    print(max_prob[k,i,j].data.numpy() * cls_prob.data.numpy())
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
  

def predict_gpu(model,image_name,root_path=''):

    image = cv2.imread(root_path+image_name)
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    

    mean = (123,117,104)#RGB
    img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    img = Variable(img[None,:,:,:])
    img = img.cuda()

    pred = model(img) #1x7x7x30
    #pred = pred.view(-1,7,7,11)
    pred = pred.cpu()
    bbox = decoder_(pred)

    return bbox
 
def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    import math
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    
    for i in range(len(boxes)):
        box = boxes[i]

        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        
        cls_id = int(box[5])
        
        img = cv2.putText(img, class_names[cls_id], (x1,y1),  cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), thickness=1)
        img = cv2.putText(img, str(round(box[4],3)), (x1,y1+20),  cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), thickness=1)
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (255, 205, 51), 2)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img,len(boxes)


if __name__ == '__main__':

    import os

    model = YOLO(1,3,False)
    
    model.load_state_dict(torch.load('./models/model_.pkl98'))
    model.cuda()
    test_path = list(map(lambda x: os.path.join('./test_images', x), os.listdir('./test_images')))
    for image_name in test_path:
        save_name = image_name.split('/')[-1]
        image = cv2.imread(image_name)
        boxes = predict_gpu(model,image_name)
        image,num = plot_boxes_cv2(image, boxes, savename='samples/'+save_name, class_names=classes, color=None)
        cv2.namedWindow('result')
        cv2.imshow('result',image)
        cv2.waitKey()

    
    
    