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

import os
import config
from torch.utils import data
from voc_eval import _do_python_eval_quite

from box_utils import yolo_box_decoder

import datasets
from torchvision import transforms


import time

def eval_mAp(model, prefix, outfile, test_list):
    res_prefix = prefix + '/' + outfile
    test_result(model, prefix, outfile, test_list)
    # _do_python_eval(res_prefix, output_dir = 'output')
    result = _do_python_eval_quite(res_prefix, output_dir='output')

    return result

def test_result(model, prefix, outfile, test_list):
    class_num = config.cls_num
    list_file = 'VOC2007_test.txt'


    class_num = 20
    transform = transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize([0.482, 0.459, 0.408], [1, 1, 1] )#[0.485, 0.456, 0.406] [1, 1, 1]
    ])

    dataset = datasets.VOCDatasets(transform,list_file ,None,False)

    test_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4)

    test_loader = tqdm(test_loader)

    list_file = test_list
    lines = []
    with open(list_file) as f:
        lines = f.readlines()

    fps = [0] * class_num
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(class_num):
        buf = '%s/%s%s.txt' % (prefix, outfile, config.classes[i])
        fps[i] = open(buf, 'w')

    for lineId, (images, label) in enumerate(test_loader):
        t1 = time.time()
        img = cv2.imread(lines[lineId].split(" ")[0])
        width, height = img.shape[1], img.shape[0]


        images = images.cuda()
        t2 = time.time()
        #print("load times: ",t2-t1)
        
        t1 = time.time()
        pred = model(images)
        t2 = time.time()
        #print("pred times: ",t2-t1)
        
        t1 = time.time()
        pred_boxes,pred_conf = yolo_box_decoder(pred)
        t2 = time.time()
        #print("decoder times: ",t2-t1)
        
        fileId = os.path.basename(lines[lineId]).split('.')[0]


        for j in range(len(pred_boxes)):


            x1 = pred_boxes[j,0]
            y1 = pred_boxes[j,1]
            x2 = x1 + pred_boxes[j,2]
            y2 = y1 + pred_boxes[j,3]

            x1,x2 = x1*width,x2*width
            y1,y2 = y1*height,y2*height

            cls_id = int(pred_conf[j,0])
            scores = pred_conf[j,1]#.max()
            #print(cls_id,scores)
            fps[cls_id].write('%s %.3f %.1f %.1f %.1f %.1f\n' %
                              (fileId, scores, x1 + 1, y1 + 1, x2 + 1, y2 + 1))
            

    for i in range(class_num):
        fps[i].close()

    return


if __name__ == '__main__':
    model = YOLO(config.cls_num, config.bbox_num, config.box_scale, conv_model=config.use_conv)
    model.load_state_dict(torch.load('./54con.pkl'))



    model.cuda()
    model.eval()
    prefix = 'results'
    outfile = "voc"
    test_list = 'VOC2007_test.txt'
    result = eval_mAp(model, prefix, outfile, test_list)

    for key, v in result.items():
        print("{} : {:.3f}".format(key, v))
    print('~~~~~~~')
    print("mean ap : {:.3f}".format(np.mean(list(result.values()))))


