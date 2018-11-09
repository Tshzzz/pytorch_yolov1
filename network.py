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

layer_configs = [
        # Unit1 (2)
        (32, 3, True),
        (64, 3, True),
        # Unit2 (3)
        (128, 3, False),
        (64, 1, False),
        (128, 3, True),
        # Unit3 (3)
        (256, 3, False),
        (128, 1, False),
        (256, 3, True),
        # Unit4 (5)
        (512, 3, False),
        (256, 1, False),
        (512, 3, False),
        (256, 1, False),
        (512, 3, True),
        # Unit5 (5)
        (1024, 3, False),
        (512, 1, False),
        (1024, 3, False),
        (512, 1, False),
        (1024, 3, False),
]


class conv_block(nn.Module):
    
    def __init__(self,inplane,outplane,kernel_size,pool,stride=1):
        super(conv_block, self).__init__()
        
        pad = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(inplane, outplane, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.act = nn.LeakyReLU(0.1)
        self.pool = pool #MaxPool2d(2,stride = 2)
        
    def forward(self,x):    
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        
        if self.pool:
            out = F.max_pool2d(out,kernel_size=2,stride=2)
            
        return out


class darknet_19(nn.Module):
    
    def __init__(self,cls_num = 1000):
        super(darknet_19, self).__init__()
        self.class_num = cls_num
        self.feature = self.make_layers(3,layer_configs)

            
    def make_layers(self,inplane,cfg):
        layers = []

        for outplane,kernel_size,pool in cfg:
            layers.append(conv_block(inplane,outplane,kernel_size,pool))
            inplane = outplane
            
            
        return nn.Sequential(*layers)
          
    def load_weight(self,weight_file):
        print("Load pretrained models !")

        fp = open(weight_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        header = torch.from_numpy(header)
        buf = np.fromfile(fp, dtype = np.float32)
        
        start = 0
        for idx,m in enumerate(self.feature.modules()):
            if isinstance(m, nn.Conv2d):
                conv = m
            if isinstance(m, nn.BatchNorm2d):
                bn = m
                start = load_conv_bn(buf,start,conv,bn)

        assert start == buf.shape[0]
        
    def forward(self, x):
        
        output = self.feature(x)
        
        return output    
    
class YOLO(nn.Module):

    def __init__(self,cls_num,bbox_num = 2,scale_size = 7,conv_model = True,if_pretrain = True):
        super(YOLO, self).__init__()
        
        self.cls_num = cls_num
        self.feature = darknet_19()
        self.conv_model = conv_model
        if if_pretrain:
            self.feature.load_weight('models/darknet19_448.conv.23')
            
            
        self.scale_size = scale_size
        self.bbox_num = bbox_num 
        self.last_output = (5*self.bbox_num+self.cls_num)

        self.boundary_class = self.scale_size * self.scale_size * self.cls_num
        self.boundary_boxes = self.boundary_class + self.scale_size * self.scale_size * self.bbox_num * 4    


        self.local_layer = nn.Sequential()
        
        self.local_layer.add_module('block_1',conv_block(1024,1024,3,False,2))
        
        self.local_layer.add_module('block_2',conv_block(1024,1024,3,False,1))
        
        self.local_layer.add_module('block_3',conv_block(1024,1024,3,False,1))
        
        self.local_layer.add_module('block_4',conv_block(1024,1024,3,False,1))
        

        self.reg_layer = nn.Sequential()
        if self.conv_model:
            self.reg_layer.add_module('local_3',nn.Conv2d(1024,self.last_output , 1, stride=1, padding=0, bias=False))
            self.reg_layer.add_module('local_bn_3', nn.BatchNorm2d(self.last_output))
        else:
            self.reg_layer.add_module('local_layer', nn.Linear(1024*7*7, 4096))
            self.reg_layer.add_module('leaky_local', nn.LeakyReLU(0.1, inplace=True))
            self.reg_layer.add_module('dropout', nn.Dropout(0.2) )
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

        output = output.view(-1,self.bbox_num*5+self.cls_num,self.scale_size,self.scale_size)

        output = output.permute(0,2,3,1).contiguous()
        
        pred_cls = output[:,:,:,:self.cls_num]
        pred_bbox = torch.cat([output[:,:,:,self.cls_num:self.cls_num+4],output[:,:,:,self.cls_num+5:self.cls_num+4+5]],-1)
        
        pred_response = torch.cat([output[:,:,:,self.cls_num+4:self.cls_num+4+1],output[:,:,:,self.cls_num+4+5:self.cls_num+4+5+1]],-1)
        
        assert self.cls_num+4+5+1 == output.size(-1)
        
        for i in range(self.scale_size):       #yyyyyyyyyy
            for j in range(self.scale_size): #xxxxxx
                pred_bbox[:,i,j,0] += float(j) / self.scale_size
                pred_bbox[:,i,j,1] += float(i) / self.scale_size

        return pred_cls,pred_response,pred_bbox
        


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
    
    net = YOLO(20,conv_model=False)

    c,r,bb = net(torch.randn(2,3,448,448))
    
    
    print(c.size()) 
    print(r.size())
    print(bb.size())
        
    
    
    
    
    
    