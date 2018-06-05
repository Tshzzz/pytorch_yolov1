#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:35:57 2017

@author: vl-tshzzz
"""
import torch
import os
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image,ImageEnhance,ImageOps
 
import random
import cv2
from torch.autograd import Variable


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#classes = ["person"]

class CustomDataset(data.Dataset):
    def __init__(self,root,transform,list_file,train):

        self.root = root
        
        self.transform = transform
        
        self.train = train
    
        self.label_path = []

        self.image_path = []
        
        self.img_size = 448
        
        self.boxes_num = 3
        
        self.ceil_size = 7
        
        self.class_num = len(classes)
        
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split(' ')
            self.image_path.append(splited[0])
            self.label_path.append(splited[1])
            

    def __getitem__(self, idx):
        
        fname = self.image_path[idx]
        gt_path = self.label_path[idx]
        #org_img = cv2.imread(fname,1)
        #cv2.imshow("dd",org_img)
        
        #cv2.waitKey()
        img = Image.open(fname)
        img = img.resize((int(self.img_size), int(self.img_size)))
        max_boxes = self.ceil_size*self.ceil_size
        

        target = np.zeros((self.ceil_size,self.ceil_size,self.boxes_num*5+self.class_num))

        bs = np.loadtxt(gt_path,delimiter=',') 
        bs = np.reshape(bs, (-1, 5))
        
        cc = 0

        dw,dy = 0,0
        if self.train == True:
            img,bs = data_augment(img,bs)
            dw = random.uniform(-0.03,0.03)
            dy = random.uniform(-0.03,0.03)


        for i in range(bs.shape[0]):

            local_x = int(bs[i][1]  * self.ceil_size ) 
            local_y = int(bs[i][2]  * self.ceil_size ) 
            
            for j in range(self.boxes_num):

                target[local_y,local_x,5*j + 4] = 1
                target[local_y,local_x,0+j*5:2+j*5] = bs[i,1:3]
                
                target[local_y,local_x,0 + j*5] = bs[i,1]
                target[local_y,local_x,1 + j*5] = bs[i,2]
                
                target[local_y,local_x,j*5 + 2] = bs[i,3] + dw
                target[local_y,local_x,j*5 + 3] = bs[i,4] + dy

            target[local_y,local_x,int(bs[i][0])+self.boxes_num*5] = 1
                
            cc += 1
            if cc >= max_boxes:
                break

          
        image = self.transform(img)

        target = torch.from_numpy(target)

        return image,target#,org_img

    def __len__(self):
        return self.num_samples
    
    
def random_contract(Coeffi,img):
    return ImageEnhance.Contrast(img).enhance(Coeffi)
        
def random_Brightness(Coeffi,img):
    return ImageEnhance.Brightness(img).enhance(Coeffi)
    
def random_Color(Coeffi,img):
    return ImageEnhance.Color(img).enhance(Coeffi) 

def random_Sharpness(Coeffi,img):
    return ImageEnhance.Sharpness(img).enhance(Coeffi) 

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augment(image,bb):
    new_bb = bb
    #coe = [0.25,0.2,0.3,0.3,0.5,0.5,0.8,0.9,1,1,1,1,1.5,1.8,1.8,2,2,2.5,3]
    #print(bb)
    if random.randint(0,100) < 50: 
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        new_bb[:,1] = 1 - bb[:,1]
        
    image = random_distort_image(image, hue=0.1, saturation=1.5, exposure=1.5)
    '''
        if random.randint(0,100) < 40: 
            ind = random.randint(0,len(coe)-1)
            image = random_contract(coe[ind],image)

            ind = random.randint(0,len(coe)-1)
            image = random_Brightness(coe[ind],image)

            ind = random.randint(0,len(coe)-1)
            image = random_Color(coe[ind],image)
    
            ind = random.randint(0,len(coe)-1)
            image = random_Sharpness(coe[ind],image)    
    '''    
    #image.show()
    return image,new_bb



 
class testDataset(data.Dataset):
    def __init__(self,root,transform,list_file):

        self.root = root
        
        self.transform = transform
    
        self.label_path = []

        self.image_path = []
        
        self.img_size = 448
        
        self.boxes_num = 2
        
        self.ceil_size = 7
        
        self.class_num = len(classes)
        
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split(' ')
            self.image_path.append(splited[0])
            self.label_path.append(splited[1])
            

    def __getitem__(self, idx):
        
        fname = self.image_path[idx]   
        img = Image.open(fname)
        img = img.resize((int(self.img_size), int(self.img_size)))

        bs = np.loadtxt(self.label_path[idx],delimiter=',') 
        bs = np.reshape(bs, (-1, 5))
        
        target = np.zeros((len(bs),len(bs[0]) + 1))
        
        target[:,0:2] = bs[:,1:3] - 0.5*pow(bs[:,3:5],2)
        target[:,2:4] = bs[:,1:3] + 0.5*pow(bs[:,3:5],2)
        
        target[:,4] = 1
        target[:,5] = bs[:,0]
        
        
        image = self.transform(img)
        
        
        return image,target    

    def __len__(self):
        return self.num_samples
    
def get_test_loader(root,list_file, image_size, batch, num_workers=5):

    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    
    dataset = testDataset(root,transform,list_file)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader    
    
    
    

    
def get_loader(root,list_file,train, image_size, batch, num_workers=8):

    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    
    dataset = CustomDataset(root,transform,list_file,train)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch,
                                  shuffle=train,
                                  num_workers=num_workers)
    return data_loader


def test_decoder(pred,image,root_path=''):

    result = []
    boxes =  decoder_(pred)
    w,h = image.shape[1],image.shape[0]
    #print(boxes)
    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        #print(x1,y1,x2,y2,w)

        cls_index = int(box[5]) # convert LongTensor to int
        prob = box[4]
        result.append([(x1,y1),(x2,y2),classes[cls_index],prob])
    cc = 0
    return result,cc

def decoder_(pred):
    bb_num = 3
    scale_size = 7
    cls_num = len(classes)
    pred = pred.view(-1,scale_size,scale_size,bb_num*5+cls_num)
    prob = pred[:,:,:,4:bb_num*5:5]
    
    max_prob,max_prob_index = prob.max(3)
    #print(max_prob)
    cell_size = 1./scale_size
    #ceil_size = 7.
    boxes=[]

    for k in range(len(pred)):
        for i in range(7):
            for j in range(7):  
                if max_prob[k,i,j] > 0.01:
                    max_prob_index_np = max_prob_index[k,i,j]

                    bbox = pred[k , i , j , max_prob_index_np*5 : max_prob_index_np*5 + 5 + 1].contiguous()      
                    _,cls = pred[k,i,j,5*bb_num:].max(0)
                    #print(bbox)
                    bbox[0] #/= scale_size 
                    bbox[1] #/= scale_size 
                    #bbox[2] *= scale_size 
                    #bbox[3] *= scale_size 
                    bbox[5] = int(cls.numpy())
                    
                    box_xy = torch.FloatTensor(bbox.size())
                    box_xy[:2] = bbox[:2] - 0.5*pow(bbox[2:4],2)
                    box_xy[2:4] = bbox[:2] + 0.5*pow(bbox[2:4],2)
                    box_xy[4] = bbox[4]
                    box_xy[5] = bbox[5]
                    boxes.append(box_xy.view(1,6).numpy()[0].tolist())
    #boxes = nms(boxes, 0.5)
    return boxes
  

if __name__ == '__main__':
    
 
    test_loader =  get_loader('./','train_list/VOC2007_train.txt',False,448,1,1)

    count = 1.

    for images,label,org_img in test_loader:

        image = org_img[0].numpy()
        image = cv2.flip(image, 1)
        result,cc = test_decoder(label,image)

        for left_up,right_bottom,class_name,prob in result:
            cv2.rectangle(image,left_up,right_bottom,(0,255,0),2)
            cv2.putText(image,class_name,left_up,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)

        h,w,_ = image.shape
        step_w = int(w / 7.)
        step_h = int(h / 7.)
        for i in range(7):
            cv2.line(image,(i*step_w,0),(i*step_w,h),(255,0,0),2)
            cv2.line(image,(0,i*step_h),(w,i*step_h),(255,0,0),2)
        cv2.namedWindow('result')
        cv2.imshow('result',image)
        cv2.waitKey()

