#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

import torchvision.transforms as transforms
import cv2
from network import YOLO
from util import decoder_#,classes
import config

def predict_gpu(model,image_name,root_path=''):

    image = cv2.imread(root_path+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h,w,_ = image.shape
    img = cv2.resize(image,(448,448)) 
    


    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = transform(img)
    img = Variable(img[None,:,:,:])
    img = img.cuda()

    pred_cls,pred_response,pred_bboxes = model(img)

    bbox = decoder_(pred_cls.cpu(),pred_response.cpu(),pred_bboxes.cpu())

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

    model = YOLO(config.cls_num,config.bbox_num,config.box_scale,conv_model = config.use_conv)
    model.eval()
    model.load_state_dict(torch.load('./runs/model_.pkl'))
    model.cuda()
    
    
    test_path = list(map(lambda x: os.path.join('./test', x), os.listdir('./test')))
    for image_name in test_path:
        save_name = image_name.split('/')[-1]
        image = cv2.imread(image_name)
        
        
        boxes = predict_gpu(model,image_name)
        image,num = plot_boxes_cv2(image, boxes, savename='samples/'+save_name, class_names=config.classes, color=None)
        cv2.namedWindow('result')
        cv2.imshow('result',image)
        cv2.waitKey()

    
    
    