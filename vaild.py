#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:05:18 2018

@author: vl-tshzzz
"""

import torch
from tqdm import tqdm
import cv2
import numpy as np
from network import YOLO
import voc_datasets
from util import decoder_vaild,decoder_,bbox_iou
import os
import config


from voc_eval import _do_python_eval_quite

def eval_f1(model,loader,iou_thresh=0.5):
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    eps = 1e-5
    ap = []
    for images,label in loader:
        label = label.view(-1,6)
        images = images.cuda()
        
        pred_cls,pred_response,pred_bboxes = model(images) #1x7x7x30
    
        pred_boxes = decoder_(pred_cls.cpu(),pred_response.cpu(),pred_bboxes.cpu())
        pred_boxes = np.array(pred_boxes)
    
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

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    
    #print("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
    return fscore,recall,precision



def eval_mAp(model,prefix,outfile,test_list):

    res_prefix = prefix +'/'+outfile
    fscore,recall,precision = test_result(model,prefix,outfile,test_list)
    #_do_python_eval(res_prefix, output_dir = 'output')
    result = _do_python_eval_quite(res_prefix, output_dir = 'output')
    print("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))
    return result
    
    

def test_result(model,prefix,outfile,test_list):
    
    class_num = config.cls_num
    
    test_loader = voc_datasets.get_loader(test_list,448,1,False,1)
    test_loader = tqdm(test_loader)
    
    list_file = test_list
    lines = []
    with open(list_file) as f:
        lines = f.readlines()
    
    fps = [0]*class_num
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(class_num):
        buf = '%s/%s%s.txt' % (prefix, outfile, config.classes[i])
        fps[i] = open(buf, 'w')
        
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    eps = 1e-5
    iou_thresh = 0.5


    for lineId,(images,label) in enumerate(test_loader):
        label = label.view(-1,6)
        images = images.cuda()
        pred_cls,pred_response,pred_bboxes = model(images)
    
        pred_boxes = decoder_vaild(pred_cls.cpu(),pred_response.cpu(),pred_bboxes.cpu(),config.cls_num)
        pred_boxes = np.array(pred_boxes)
        
        
        pred_boxes_f1 = decoder_(pred_cls.cpu(),pred_response.cpu(),pred_bboxes.cpu())
        pred_boxes_f1 = np.array(pred_boxes_f1)
        
        fileId = os.path.basename(lines[lineId]).split('.')[0]
        
        img = cv2.imread(lines[lineId].split(" ")[0])
        width, height = img.shape[1],img.shape[0] #get_image_size(valid_files[lineId])


        num_gts = len(label)
        total = total + num_gts
        proposals += len(pred_boxes)

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
                
                
        for i in range(num_gts):
            box_gt = [label[i][0], label[i][1], label[i][2], label[i][3], label[i][4], label[i][5]]
                
            best_iou = 0
            best_j = -1
            for j in range(len(pred_boxes_f1)):
                iou = bbox_iou(box_gt, pred_boxes_f1[j], x1y1x2y2=True)
                if iou > best_iou:
                    best_j = j
                    best_iou = iou
            
            if best_iou > iou_thresh and int(pred_boxes_f1[best_j][5]) == int(box_gt[5]):
                correct = correct+1        
      
        
        
                
    for i in range(class_num):
        fps[i].close()

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)  
    
    return fscore,recall,precision


if __name__ == '__main__':

    model = YOLO(config.cls_num, config.bbox_num, config.box_scale, conv_model = config.use_conv )
    
    
    model.load_state_dict(torch.load('./runs/model_.pkl'))
    model.cuda()
    model.eval()
    prefix = 'results'
    outfile = "voc"
    test_list = 'train_list/VOC2007_test.txt'
    result = eval_mAp(model,prefix,outfile,test_list)

    for key,v in result.items():
        print("{} : {}".format(key,v))
    print('~~~~~~~')
    print("mean ap : {}".format(np.mean(list(result.values()))))
    
    
