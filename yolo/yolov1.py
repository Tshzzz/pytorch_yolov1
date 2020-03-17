# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:38:10 2018

@author: vl-tshzzz
"""

import torch
import torch.nn as nn
from yolo.decoder import yolo_decoder
from yolo.darknet import darknet_19,conv_block
from yolo.loss import yolov1_loss

def create_yolov1(cfg):
    cls_num = cfg['class_num']
    box_num = cfg['box_num']
    ceil_size = cfg['ceil_size']
    pretrained = cfg['pretrained']
    model = YOLO(cls_num,box_num,ceil_size,pretrained)

    return model


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class YOLO(nn.Module):

    def __init__(self, cls_num, bbox_num=2, scale_size=7, pretrained=None):
        super(YOLO, self).__init__()

        self.cls_num = cls_num
        self.backbone = darknet_19()
        if pretrained is not None:
            self.backbone.load_weight(pretrained)

        self.loss = yolov1_loss(2, 5, 0.5)
        self.scale_size = scale_size
        self.bbox_num = bbox_num
        self.last_output = (5 * self.bbox_num + self.cls_num)

        self.local_layer = nn.Sequential()
        self.local_layer.add_module('block_1', conv_block(1024, 1024, 3, False, 2))
        self.local_layer.add_module('block_2', conv_block(1024, 1024, 3, False, 1))
        self.local_layer.add_module('block_3', conv_block(1024, 1024, 3, False, 1))
        self.local_layer.add_module('block_4', conv_block(1024, 1024, 3, False, 1))
        fill_fc_weights(self.local_layer)

        self.reg_layer = nn.Sequential()
        self.reg_layer.add_module('local_layer', nn.Linear(1024 * 7 * 7, 4096))
        self.reg_layer.add_module('leaky_local', nn.LeakyReLU(0.1, inplace=True))
        self.reg_layer.add_module('dropout', nn.Dropout(0.5))
        fill_fc_weights(self.reg_layer)

        self.cls_pred =  nn.Linear(4096, self.cls_num * self.scale_size * self.scale_size)
        self.response_pred = nn.Linear(4096, self.bbox_num * self.scale_size * self.scale_size)
        self.offset_pred = nn.Linear(4096, self.bbox_num * 4 * self.scale_size * self.scale_size)
        fill_fc_weights(self.cls_pred)
        fill_fc_weights(self.response_pred)
        fill_fc_weights(self.offset_pred)

    def forward(self, x, target=None,conf=0.02, topk=100, nms_threshold=0.5):
        B,w,h,c = x.shape
        img_size = (w,h)
        output = self.backbone(x)
        output = self.local_layer(output).view(B,-1)
        #print(output.shape)
        output = self.reg_layer(output)
        #print(output.shape)

        pred_cls = self.cls_pred(output).view(B,self.cls_num,self.scale_size,self.scale_size)
        pred_response = self.response_pred(output).view(B,self.bbox_num,self.scale_size,self.scale_size)
        pred_bbox = self.offset_pred(output).view(B,self.bbox_num*4,self.scale_size,self.scale_size)

        if target is None:
            output = []
            for bs in range(B):
                cls = pred_cls[bs,:,:,:]
                objness = pred_response[bs,:,:,:]
                bbox = pred_bbox[bs,:,:,:]
                output.append(yolo_decoder((cls,objness,bbox),img_size,conf,nms_threshold,topk))
            return output

        else:
            pred = (pred_cls,pred_response,pred_bbox)
            loss_dict = self.loss(pred,target)
            return loss_dict



if __name__ == '__main__':
    net = YOLO(20)

    c, r, bb = net(torch.randn(2, 3, 448, 448))

    print(c.size())
    print(r.size())
    print(bb.size())






