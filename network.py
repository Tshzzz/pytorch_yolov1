#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:38:10 2018

@author: vl-tshzzz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class darknet_19(nn.Module):
    def __init__(self,pretrained,cls_num = 1000):
        super(darknet_19, self).__init__()
        
        self.pretrain = pretrained
        self.class_num = cls_num
        model = nn.Sequential()
            
        model.add_module('conv_1', nn.Conv2d(3, 64, 7, stride=2, padding=1, bias=False))
        model.add_module('bn_1', nn.BatchNorm2d(64))
        model.add_module('leaky_1', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_1',nn.MaxPool2d( 2,stride=2))



        model.add_module('conv_2', nn.Conv2d(64, 192, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_2', nn.BatchNorm2d(192))
        model.add_module('leaky_2', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_2',nn.MaxPool2d( 2,stride=2))


        model.add_module('conv_3', nn.Conv2d(192, 128, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_3', nn.BatchNorm2d(128))
        model.add_module('leaky_3', nn.LeakyReLU(0.1, inplace=True))
    
        model.add_module('conv_4', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_4', nn.BatchNorm2d(256))
        model.add_module('leaky_4', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_5', nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_5', nn.BatchNorm2d(256))
        model.add_module('leaky_5', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_6', nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_6', nn.BatchNorm2d(512))
        model.add_module('leaky_6', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('maxpool_3',nn.MaxPool2d( 2,stride=2))
        
        
        model.add_module('conv_7', nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_7', nn.BatchNorm2d(256))
        model.add_module('leaky_7', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_8', nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_8', nn.BatchNorm2d(512))
        model.add_module('leaky_8', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_9', nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_9', nn.BatchNorm2d(256))
        model.add_module('leaky_9', nn.LeakyReLU(0.1, inplace=True))
            
        model.add_module('conv_10', nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_10', nn.BatchNorm2d(512))
        model.add_module('leaky_10', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_11', nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_11', nn.BatchNorm2d(256))
        model.add_module('leaky_11', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_12', nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_12', nn.BatchNorm2d(512))
        model.add_module('leaky_12', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_13', nn.Conv2d(512, 256, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_13', nn.BatchNorm2d(256))
        model.add_module('leaky_13', nn.LeakyReLU(0.1, inplace=True))
            
        model.add_module('conv_14', nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_14', nn.BatchNorm2d(512))
        model.add_module('leaky_14', nn.LeakyReLU(0.1, inplace=True))
    
        model.add_module('conv_15', nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_15', nn.BatchNorm2d(512))
        model.add_module('leaky_15', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_16', nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_16', nn.BatchNorm2d(1024))
        model.add_module('leaky_16', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('maxpool_4',nn.MaxPool2d(2, stride=2))
            
            
        model.add_module('conv_17', nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_17', nn.BatchNorm2d(512))
        model.add_module('leaky_17', nn.LeakyReLU(0.1, inplace=True))
     
        model.add_module('conv_18', nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_18', nn.BatchNorm2d(1024))
        model.add_module('leaky_18', nn.LeakyReLU(0.1, inplace=True))
        
        model.add_module('conv_19', nn.Conv2d(1024, 512, 1, stride=1, padding=0, bias=False))
        model.add_module('bn_19', nn.BatchNorm2d(512))
        model.add_module('leaky_19', nn.LeakyReLU(0.1, inplace=True))     
        
        model.add_module('conv_20', nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_20', nn.BatchNorm2d(1024))
        model.add_module('leaky_20', nn.LeakyReLU(0.1, inplace=True))  
        
#==============    
        
        self.models = model

        if self.pretrain:
            self.fc = nn.Linear(1024, self.class_num)
            
            
    def load_weight(self,weight_file):
        print("Load pretrained models !")
        fp = open(weight_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        #print(header)
        header = torch.from_numpy(header)
        seen = header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()
        load_weight_list = []
        layer_index = 0
        for layer,module in self.models.named_children():
            layer_type = layer.split('_')[0]
            if layer_type == 'conv':
                load_weight_list.append(layer_index)
            layer_index += 1
        start = 0
        count = 1
        for conv_index in load_weight_list:   
            bn_index = conv_index + 1
            start = load_conv_bn(buf,start,self.models[conv_index] , self.models[bn_index] )
            count += 1

        
    def forward(self, x):
        output = self.models(x)
        
        if self.pretrain:
            output = F.avg_pool2d(output, (output.size(2), output.size(3)))
            output = output.squeeze()           
            output = F.softmax(self.fc(output))
        
        return output    
    
class YOLO(nn.Module):

    def __init__(self,cls_num,bbox_num = 2,conv_model = True,if_pretrain = True):
        super(YOLO, self).__init__()
        
        self.cls_num = cls_num
        self.feature = darknet_19(False)
        self.conv_model = conv_model
        if if_pretrain:
            self.feature.load_weight('models/pretrain_model/extraction.conv.weights')
            
            
        self.scale_size = 7
        self.bbox_num = bbox_num 
        self.last_output = (5*self.bbox_num+self.cls_num)

        

        
        self.local_layer = nn.Sequential()
        self.local_layer.add_module('conv_21', nn.Conv2d(1024, 1024, 3, stride=2, padding=1, bias=False))
        self.local_layer.add_module('bn_21', nn.BatchNorm2d(1024))
        self.local_layer.add_module('leaky_21', nn.LeakyReLU(0.1, inplace=True))  
        
        self.local_layer.add_module('conv_22', nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=False))
        self.local_layer.add_module('bn_22', nn.BatchNorm2d(1024))
        self.local_layer.add_module('leaky_22', nn.LeakyReLU(0.1, inplace=True))  
        
        self.local_layer.add_module('conv_23', nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=False))
        self.local_layer.add_module('bn_23', nn.BatchNorm2d(1024))
        self.local_layer.add_module('leaky_23', nn.LeakyReLU(0.1, inplace=True))  
        
        self.local_layer.add_module('conv_24', nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=False))
        self.local_layer.add_module('bn_24', nn.BatchNorm2d(1024))
        self.local_layer.add_module('leaky_24', nn.LeakyReLU(0.1, inplace=True))   
        
        #7*7
        self.reg_layer = nn.Sequential()
        if self.conv_model:
            self.reg_layer.add_module('local_3',nn.Conv2d(1024,self.last_output , 3, stride=1, padding=1, bias=False))
            self.reg_layer.add_module('local_bn_3', nn.BatchNorm2d(self.last_output))
        else:
            self.reg_layer.add_module('local_layer',nn.Linear(1024*7*7, 4096))
            self.reg_layer.add_module('leaky_local', nn.LeakyReLU(0.1, inplace=True))
            self.reg_layer.add_module('dropout', nn.Dropout(0.5) )
            self.reg_layer.add_module('fc_1', nn.Linear(4096, (5*self.bbox_num+self.cls_num)*49 ))
        
        
    def forward(self,x):

        output = self.feature(x)
        
        output = self.local_layer(output)

        if self.conv_model:
            output = self.reg_layer(output)
            output = output.permute(0,2,3,1).contiguous()
        else:
            output = output.view(output.data.size(0),-1)
            output = self.reg_layer(output)

        output = output.view(-1,7,7,self.bbox_num*5+self.cls_num)
        
        
        for i in range(len(output[0])):       #yyyyyyyyyy
            for j in range(len(output[0][0])): # xxxxxx
                output[:,i,j,0] += float(j) / self.scale_size
                output[:,i,j,1] += float(i) / self.scale_size

        return output


class TinyYOLO(nn.Module):

    def __init__(self,cls_num,bbox_num = 2,if_pretrain = True):
        super(TinyYOLO, self).__init__()
        
        self.pretrain = if_pretrain
        self.class_num = cls_num
        model = nn.Sequential()
            
        model.add_module('conv_1', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_1', nn.BatchNorm2d(16))
        model.add_module('leaky_1', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_1',nn.MaxPool2d( 2,stride=2))


        model.add_module('conv_2', nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_2', nn.BatchNorm2d(32))
        model.add_module('leaky_2', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_2',nn.MaxPool2d( 2,stride=2))
        
        model.add_module('conv_3', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_3', nn.BatchNorm2d(64))
        model.add_module('leaky_3', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_3',nn.MaxPool2d( 2,stride=2))
                
        model.add_module('conv_4', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_4', nn.BatchNorm2d(128))
        model.add_module('leaky_4', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_4',nn.MaxPool2d( 2,stride=2))
                
        
        model.add_module('conv_5', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_5', nn.BatchNorm2d(256))
        model.add_module('leaky_5', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_5',nn.MaxPool2d( 2,stride=2))

        model.add_module('conv_6', nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_6', nn.BatchNorm2d(512))
        model.add_module('leaky_6', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_6',nn.MaxPool2d( 2,stride=2))      
        
        model.add_module('conv_7', nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_7', nn.BatchNorm2d(1024))
        model.add_module('leaky_7', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_7',nn.MaxPool2d( 2,stride=2))
        
        model.add_module('conv_8', nn.Conv2d(1024, 256, 3, stride=1, padding=1, bias=False))
        model.add_module('bn_8', nn.BatchNorm2d(256))
        model.add_module('leaky_8', nn.LeakyReLU(0.1, inplace=True))
        model.add_module('maxpool_8',nn.MaxPool2d( 2,stride=2))        
        
        
        self.reg_layer = nn.Sequential()
        self.reg_layer.add_module('dropout', nn.Dropout(0.5) )
        self.reg_layer.add_module('fc_1', nn.Linear(7*7*256, (5*self.bbox_num+self.cls_num)*49 ))
        
        self.models = model

        
    def forward(self,x):

        output = self.models(x)
        
        print(output.size())
        output = output.view(output.data.size(0),-1)
        
        output = self.reg_layer(output)
        
        output = output.view(-1,7,7,self.bbox_num*5+self.cls_num)

        for i in range(len(output[0])):       #yyyyyyyyyy
            for j in range(len(output[0][0])): # xxxxxx
                output[:,i,j,0] += float(j) / self.scale_size
                output[:,i,j,1] += float(i) / self.scale_size

        return output




def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()

    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b

    
    conv_weight = torch.from_numpy(buf[start:start+num_w])
    conv_model.weight.data.copy_(conv_weight.view_as(conv_model.weight)); start = start + num_w 
    return start



if __name__ == '__main__':
    a = TinyYOLO()