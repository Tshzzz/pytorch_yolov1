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

    def __init__(self, inplane, outplane, kernel_size, pool, stride=1):
        super(conv_block, self).__init__()

        pad = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(inplane, outplane, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.act = nn.LeakyReLU(0.1)
        self.pool = pool  # MaxPool2d(2,stride = 2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        if self.pool:
            out = F.max_pool2d(out, kernel_size=2, stride=2)

        return out


class darknet_19(nn.Module):

    def __init__(self, cls_num=1000):
        super(darknet_19, self).__init__()
        self.class_num = cls_num
        self.feature = self.make_layers(3, layer_configs)

    def make_layers(self, inplane, cfg):
        layers = []

        for outplane, kernel_size, pool in cfg:
            layers.append(conv_block(inplane, outplane, kernel_size, pool))
            inplane = outplane

        return nn.Sequential(*layers)

    def load_weight(self, weight_file):
        print("Load pretrained models !")

        fp = open(weight_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        header = torch.from_numpy(header)
        buf = np.fromfile(fp, dtype=np.float32)

        start = 0
        for idx, m in enumerate(self.feature.modules()):
            if isinstance(m, nn.Conv2d):
                conv = m
            if isinstance(m, nn.BatchNorm2d):
                bn = m
                start = load_conv_bn(buf, start, conv, bn)

        assert start == buf.shape[0]

    def forward(self, x):

        output = self.feature(x)

        return output


class YOLO(nn.Module):

    def __init__(self, cls_num, bbox_num=2, scale_size=7, conv_model=True, pretrained=None):
        super(YOLO, self).__init__()

        self.cls_num = cls_num
        self.feature = darknet_19()
        self.conv_model = conv_model
        if pretrained is not None:
            self.feature.load_weight(pretrained)

        self.scale_size = scale_size
        self.bbox_num = bbox_num
        self.last_output = (5 * self.bbox_num + self.cls_num)

        cx = torch.linspace(0.5 / scale_size, (scale_size - 0.5) / scale_size, steps=scale_size). \
            view(-1, scale_size).repeat((scale_size, 1)).view(scale_size, scale_size, -1)
        cy = torch.linspace(0.5 / scale_size, (scale_size - 0.5) / scale_size, steps=scale_size). \
            view(scale_size, -1).repeat((1, scale_size)).view(scale_size, scale_size, -1)
        self.anchor = torch.cat((cx, cy), 2)

        self.local_layer = nn.Sequential()

        self.local_layer.add_module('block_1', conv_block(1024, 1024, 3, False, 2))

        self.local_layer.add_module('block_2', conv_block(1024, 1024, 3, False, 1))

        self.local_layer.add_module('block_3', conv_block(1024, 1024, 3, False, 1))

        self.local_layer.add_module('block_4', conv_block(1024, 1024, 3, False, 1))

        self.reg_layer = nn.Sequential()
        if self.conv_model:
            self.reg_layer.add_module('local_3', nn.Conv2d(1024, self.last_output, 1, stride=1, padding=0, bias=False))
            self.reg_layer.add_module('local_bn_3', nn.BatchNorm2d(self.last_output))
        else:
            self.reg_layer.add_module('local_layer', nn.Linear(1024 * 7 * 7, 4096))
            self.reg_layer.add_module('leaky_local', nn.LeakyReLU(0.1, inplace=True))
            self.reg_layer.add_module('dropout', nn.Dropout(0.5))
            #self.reg_layer.add_module('fc_1', nn.Linear(4096, (
            #            5 * self.bbox_num + self.cls_num) * self.scale_size * self.scale_size))

            self.cls_pred =  nn.Linear(4096, self.cls_num * self.scale_size * self.scale_size)
            self.response_pred = nn.Linear(4096, self.bbox_num * self.scale_size * self.scale_size)
            self.offset_pred = nn.Linear(4096, self.bbox_num * 4 * self.scale_size * self.scale_size)



    def decoder(self,pred,conf=0.02,topk=100,nms_threshold=0.5):

        '''
        pred_cls = [C,S,S]
        pred_response = [2,S,S]
        pred_bboxes = [4*2,S,S]
        '''
        pred_cls, pred_response, pred_bboxes = pred

        class_num,h,w = pred_cls.shape


        x_list, y_list, c_list = self.get_kp_torch(pred_cls, conf=conf, topk=topk)
        cls, idx = torch.sort(c_list)

        bboxes = []
        scores = []
        labels = []

        for i in range(class_num):
            mask = idx[cls.eq(i)]
            if len(mask) > 0:
                y = y_list[mask]
                x = x_list[mask]
                c = c_list[mask]
                cls_score = pred_cls[c, y, x]
                box_idx = pred_response[:,y,x].argmax()

                score = pred_response[box_idx,y,x] * cls_score

                mask_score = score > conf

                if mask_score.sum() <= 0:
                    continue

                score = score[mask_score]
                x = x[mask_score]
                y = y[mask_score]

                offsets = pred_bboxes[box_idx:box_idx+4,:,:]

                ox,oy,bw,bh = offsets

                #ceil center
                cx = (x+0.5) + ox
                cy = (y+0.5) + oy
                bw = bw * bw
                bh = bh * bh

                x1 = torch.clamp(cx - bw / 2, 0, w).unsqueeze(dim=1)
                y1 = torch.clamp(cy - bh / 2, 0, h).unsqueeze(dim=1)
                x2 = torch.clamp(cx + bw / 2, 0, w).unsqueeze(dim=1)
                y2 = torch.clamp(cy + bh / 2, 0, h).unsqueeze(dim=1)

                bbox = torch.cat((x1, y1, x2, y2), dim=1)
                keep = nms(bbox, score, nms_threshold)
                scores += score[keep].tolist()
                labels += [i for _ in range(len(keep))]
                bboxes += bbox[keep].tolist()


        if len(bboxes) > 0:
            scores = np.asarray(scores)
            bboxes = np.asarray(bboxes)
            labels = np.asarray(labels)

            box = BoxList(bboxes, (w, h))
            box.add_field('scores', scores)
            box.add_field('labels', labels)
        else:
            box = BoxList(np.asarray([[0., 0., 1., 1.]]), (w, h))
            box.add_field('scores', np.asarray([0.]))
            box.add_field('labels', np.asarray([0.]))

        return box



    def forward(self, x, target):
        B = x.size(0)
        output = self.feature(x)
        output = self.local_layer(output)
        output = self.reg_layer(output)


        output = output.view(output.data.size(0), -1)
        output = self.reg_layer(output)


        pred_cls = self.cls_pred(output).view(B,self.cls_num,self.scale_size,self.scale_size)
        pred_response = self.response_pred(output).view(B,self.bbox_num,self.scale_size,self.scale_size)
        pred_bbox = self.offset_pred(output).view(B,self.bbox_num*4,self.scale_size,self.scale_size)


        return pred_cls, pred_response, pred_bbox


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()

    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b

    conv_weight = torch.from_numpy(buf[start:start + num_w])
    conv_model.weight.data.copy_(conv_weight.view_as(conv_model.weight))
    start = start + num_w
    return start


if __name__ == '__main__':
    net = YOLO(20, conv_model=False)

    c, r, bb = net(torch.randn(2, 3, 448, 448))

    print(c.size())
    print(r.size())
    print(bb.size())






