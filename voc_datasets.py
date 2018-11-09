#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:35:57 2017

@author: vl-tshzzz
"""
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image
import config
from data_augment import load_data_detection

class TrainDatasets(data.Dataset):
    def __init__(self,transform,img_size,list_file):

        self.transform = transform
        
        self.label_path = []
        self.image_path = []
        self.img_size = img_size
        self.boxes_num = config.bbox_num
        self.ceil_size = config.box_scale
        self.class_num = config.cls_num
        
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

        max_boxes = self.ceil_size*self.ceil_size
        
        bb_class = np.zeros((self.ceil_size,self.ceil_size,self.class_num))
        bb_response = np.zeros((self.ceil_size,self.ceil_size,self.boxes_num))
        bb_boxes = np.zeros((self.ceil_size,self.ceil_size,4*self.boxes_num))

        cc = 0
        
        img, bs = load_data_detection(fname,gt_path, (self.img_size,self.img_size), 
                                      config.jitter,  config.hue,  config.saturation,  config.exposure)

        for i in range(bs.shape[0]):

            local_x = int(bs[i][1]  * self.ceil_size ) 
            local_y = int(bs[i][2]  * self.ceil_size ) 
            
            for j in range(self.boxes_num):
                
                bb_response[local_y,local_x,j] = 1      
                
                bb_boxes[local_y,local_x,j*4 + 0] = bs[i,1]
                bb_boxes[local_y,local_x,j*4 + 1] = bs[i,2]
                
                bb_boxes[local_y,local_x,j*4 + 2] = np.sqrt(bs[i,3])
                bb_boxes[local_y,local_x,j*4 + 3] = np.sqrt(bs[i,4])
                assert bs[i,3] >= 0 
                assert bs[i,4] >= 0 
            
            
            bb_class[local_y,local_x,int(bs[i][0])] = 1

                
            cc += 1
            if cc >= max_boxes:
                break
  
        image = self.transform(img)
        bb_class = torch.from_numpy(bb_class)
        bb_response = torch.from_numpy(bb_response)
        bb_boxes = torch.from_numpy(bb_boxes)

        return image,bb_class,bb_response,bb_boxes,fname

    def __len__(self):
        return self.num_samples
    
 
class testDataset(data.Dataset):
    def __init__(self,transform,img_size,list_file):
        
        self.transform = transform
    
        self.label_path = []

        self.image_path = []
        
        self.img_size = img_size
        
        
        self.ceil_size = config.box_scale
        
        self.class_num = config.cls_num
        
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split(' ')
            self.image_path.append(splited[0])
            self.label_path.append(splited[1])
            

    def __getitem__(self, idx):
        
        fname = self.image_path[idx]   
        img = Image.open(fname).convert('RGB')
        img = img.resize((int(self.img_size), int(self.img_size)))

        bs = np.loadtxt(self.label_path[idx],delimiter=',') 
        bs = np.reshape(bs, (-1, 5))

        target = np.zeros((len(bs),len(bs[0]) + 1))
        
        target[:,0:2] = bs[:,1:3] - 0.5*bs[:,3:5]
        target[:,2:4] =  bs[:,1:3] + 0.5*bs[:,3:5]
        
        target[:,4] = 1
        target[:,5] = bs[:,0]
        
        
        image = self.transform(img)
        
        
        return image,target    

    def __len__(self):
        return self.num_samples
    
def get_loader(list_file, image_size, batch,train=True, num_workers=8):

    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
    if train:
        dataset = TrainDatasets(transform,image_size,list_file)
    else:
        dataset = testDataset(transform,image_size,list_file)
        
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch,
                                  shuffle=train,
                                  num_workers=num_workers)
    
    return data_loader

if __name__ == '__main__':
    from PIL import ImageDraw
    from tqdm import tqdm
    #test_loader =  get_loader('train_list/show_list.txt',448,1,True,1)
    
    test_loader =  get_loader('train_list/show_list.txt',448,1,False,8)
    count = 0
    test_loader = tqdm(test_loader)
    unloader = transforms.ToPILImage()
    
    #for img,pred_cls,pred_response,pred_bboxes,_ in test_loader:
    for img,label in test_loader:
        #print(pred_cls.shape)
        #print(pred_bboxes.shape)
        #print(pred_response.shape)
        #pred_cls.cpu(),pred_response.cpu(),pred_bboxes.cpu()
        #label = util.decoder_(pred_cls,pred_response,pred_bboxes)
        
        #print(label[0].shape)
        #print(label.shape)
        label = label.view(-1,6)
        
        img = img.cpu().clone().squeeze(0)
        image= unloader(img)
        #image.show()
        num_gts = len(label)
        drawObject = ImageDraw.Draw(image)
        #print(num_gts)
        for i in range(num_gts):
            #print(label[i][6])
            if label[i][5] > 0:
                box_gt = [label[i][0]*448, label[i][1]*448, label[i][2]*448, label[i][3]*448, label[i][4]*448, label[i][5]]
                
                if box_gt[0] < 0 or box_gt[1] < 0 or box_gt[2] > 448 or box_gt[3] > 448:# or label[i][5] > 20:
                    count += 1
                    #fff
                #print([(,box_gt[1]),(box_gt[2],box_gt[1])])
                
                box_gt[0] = max(1,box_gt[0])
                box_gt[2] = max(1,box_gt[2])
                
                box_gt[1] = min(447,box_gt[1])
                box_gt[3] = min(447,box_gt[3])
                
                
                drawObject.line([(box_gt[0],box_gt[1]),(box_gt[2],box_gt[1])],"red")
                drawObject.line([(box_gt[2],box_gt[1]),(box_gt[2],box_gt[3])],"red")
                drawObject.line([(box_gt[2],box_gt[3]),(box_gt[0],box_gt[3])],"red")
                drawObject.line([(box_gt[0],box_gt[3]),(box_gt[0],box_gt[1])],fill = "red")   
            

        draw = ImageDraw.Draw(image)

        #draw.rectangle((box_gt[0]*448,box_gt[1]*448,box_gt[2]*448 ,box_gt[3]*448))

        image.show()

    print(count)
