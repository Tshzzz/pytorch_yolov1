#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:01:01 2018

@author: vl-tshzzz
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class yolov1_loss(nn.Module):
    def __init__(self, B, l_coord, l_noobj,device='cuda' ,cls_num=20):
        super(yolov1_loss, self).__init__()
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.class_num = cls_num
        self.device = device

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''

        lt = torch.max(
            box1[:, :2],  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2],  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:],  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:],  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, 0] * wh[:, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self,pred,target):
        #print(target)
        pred_cls, pred_response, pred_bboxes = pred
        label_cls, label_response, label_bboxes = target


        pred_cls = pred_cls.to(self.device)
        pred_response =  pred_response.to(self.device)
        pred_bboxes = pred_bboxes.to(self.device)

        label_cls = label_cls.to(self.device)
        label_response =  label_response.to(self.device)
        label_bboxes = label_bboxes.to(self.device)



        batch_size = pred_response.size(0)

        no_obj_mask = (label_response[:, :, :, 0] < 1).unsqueeze(-1).expand_as(label_response)

        obj_response_mask = (label_response[:, :, :, 0] > 0).unsqueeze(-1).expand_as(label_response)

        obj_box_mask = (label_response[:, :, :, 0] > 0).unsqueeze(-1).expand_as(label_bboxes)

        obj_cls_mask = (label_response[:, :, :, 0] > 0).unsqueeze(-1).expand_as(label_cls)

        no_obj_contain_pred = pred_response[no_obj_mask].view(-1)
        no_obj_contain_target = label_response[no_obj_mask].view(-1)


        obj_contain_pred = pred_response[obj_response_mask].view(-1, self.B)

        obj_contain_target = label_response[obj_response_mask].view(-1, self.B)

        # class pred response
        obj_class_pred = pred_cls[obj_cls_mask].view(-1, self.class_num)
        obj_class_target = label_cls[obj_cls_mask].view(-1, self.class_num)

        # box pred response
        obj_loc_pred = pred_bboxes[obj_box_mask].view(-1, self.B * 4)
        obj_loc_target = label_bboxes[obj_box_mask].view(-1, self.B * 4)

        iou = torch.zeros(obj_loc_pred.size(0), self.B)

        for j in range(self.B):
            pred_bb = torch.zeros(obj_loc_pred.size(0), 4)

            target_bb = torch.zeros(obj_loc_pred.size(0), 4)

            target_bb[:, 0] = obj_loc_target[:, j * 4] - 0.5 * pow(obj_loc_target[:, j * 4 + 2], 2)
            target_bb[:, 1] = obj_loc_target[:, j * 4 + 1] - 0.5 * pow(obj_loc_target[:, j * 4 + 3], 2)
            target_bb[:, 2] = obj_loc_target[:, j * 4] + 0.5 * pow(obj_loc_target[:, j * 4 + 2], 2)
            target_bb[:, 3] = obj_loc_target[:, j * 4 + 1] + 0.5 * pow(obj_loc_target[:, j * 4 + 3], 2)

            pred_bb[:, 0] = obj_loc_pred[:, j * 4] - 0.5 * pow(obj_loc_pred[:, j * 4 + 2], 2)
            pred_bb[:, 1] = obj_loc_pred[:, j * 4 + 1] - 0.5 * pow(obj_loc_pred[:, j * 4 + 3], 2)
            pred_bb[:, 2] = obj_loc_pred[:, j * 4] + 0.5 * pow(obj_loc_pred[:, j * 4 + 2], 2)
            pred_bb[:, 3] = obj_loc_pred[:, j * 4 + 1] + 0.5 * pow(obj_loc_pred[:, j * 4 + 3], 2)

            iou[:, j] = self.compute_iou(target_bb, pred_bb)

        max_iou, max_index = iou.max(1)
        min_iou, _ = iou.min(1)
        max_index = max_index.data.cpu()

        coo_response_mask = torch.BoolTensor(obj_loc_pred.size(0), self.B * 4).to(self.device)

        coo_response_mask.zero_()
        for i in range(obj_loc_pred.size(0)):
            coo_response_mask[i, max_index[i] * 4:max_index[i] * 4 + 4] = 1

        obj_axis_pred = obj_loc_pred[coo_response_mask].view(-1, 4)
        obj_axis_target = obj_loc_target[coo_response_mask].view(-1, 4)

        iou_response_mask = coo_response_mask[:, [i * 4 for i in range(self.B)]]

        obj_response_pred = obj_contain_pred[iou_response_mask].view(-1)
        obj_response_target = obj_contain_target[iou_response_mask].view(-1)

        obj_local_loss = F.mse_loss(obj_axis_pred[:, 0:2], obj_axis_target[:, 0:2], reduction='sum') + \
                         F.mse_loss(obj_axis_pred[:, 2:4], obj_axis_target[:, 2:4], reduction='sum')


        obj_local_loss = obj_local_loss / batch_size
        obj_class_loss = F.mse_loss(obj_class_pred, obj_class_target, reduction='sum') / batch_size


        max_iou = (max_iou.data).to(self.device)
        obj_contain_loss = F.mse_loss(obj_response_pred, max_iou, reduction='sum') / batch_size
        no_obj_contain_loss = F.mse_loss(no_obj_contain_pred, no_obj_contain_target, reduction='sum') / batch_size
        iou_loss = F.mse_loss(max_iou, obj_response_target, reduction='sum') / batch_size


        loss_dict = {
            'offset': self.l_coord * obj_local_loss,
            'cls': obj_class_loss,
            'pObj': obj_contain_loss,
            'nObj': self.l_noobj * no_obj_contain_loss,
            'iou': iou_loss,
        }

        return loss_dict


if __name__ == '__main__':
    test_loss = yolov1_loss(2, 5, 0.5,'cuda')

    batch_size = 7

    label_cls = torch.zeros(batch_size, 7, 7, 20)
    label_bbox = torch.zeros(batch_size, 7, 7, 4 * 2)
    label_response = torch.zeros(batch_size, 7, 7, 2)

    label_response[0, 5, 3, :] = 1
    label_bbox[0, 5, 3, 0] = 0.1
    label_bbox[0, 5, 3, 1] = 0.1
    label_bbox[0, 5, 3, 2] = 0.2
    label_bbox[0, 5, 3, 3] = 0.3
    label_bbox[0, 5, 3, 4] = 0.1
    label_bbox[0, 5, 3, 5] = 0.1
    label_bbox[0, 5, 3, 6] = 0.2
    label_bbox[0, 5, 3, 7] = 0.3

    pred_cls = torch.zeros(batch_size, 7, 7, 20)
    pred_bbox = torch.zeros(batch_size, 7, 7, 4 * 2)
    pred_response = torch.zeros(batch_size, 7, 7, 2)

    pred_response[0, 5, 3, :] = 1
    pred_bbox[0, 5, 3, 0] = 0.1
    pred_bbox[0, 5, 3, 1] = 0.1
    pred_bbox[0, 5, 3, 2] = 0.2
    pred_bbox[0, 5, 3, 3] = 0.3
    pred_bbox[0, 5, 3, 4] = 0.1
    pred_bbox[0, 5, 3, 5] = 0.1
    pred_bbox[0, 5, 3, 6] = 0.2
    pred_bbox[0, 5, 3, 7] = 0.3



    pred = (pred_cls, pred_response, pred_bbox)
    target = (label_cls, label_response, label_bbox)
    test_loss(pred,target)



