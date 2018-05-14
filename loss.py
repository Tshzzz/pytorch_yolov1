# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:51:05 2018

@author: tshzzz
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:01:01 2018

@author: vl-tshzzz
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable



class yolov1_loss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj,cls_num = 1):
        super(yolov1_loss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.class_num = cls_num
        
    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''

        lt = torch.max(
            box1[:,:2],  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2],  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:],  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:],  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,0] * wh[:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]


        iou = inter / (area1 + area2 - inter)
        return iou
     
    def bbox_iou(self,box1, box2, x1y1x2y2=True):
        if x1y1x2y2:
            mx = torch.min(box1[:,0], box2[:,0])
            Mx = torch.max(box1[:,2], box2[:,2])
            my = torch.min(box1[:,1], box2[:,1])
            My = torch.max(box1[:,3], box2[:,3])
            w1 = box1[:,2] - box1[:,0]
            h1 = box1[:,3] - box1[:,1]
            w2 = box2[:,2] - box2[:,0]
            h2 = box2[:,3] - box2[:,1]

        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh
        carea = 0
        if cw <= 0 or ch <= 0:
            return 0.0
    
        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        return carea/uarea    
    def forward(self,pred,target,if_cuda = True):

        batch_size = target.size(0)

        obj_mask = (target[:,:,:,4] > 0).unsqueeze(-1).expand_as(target)
        no_obj_mask = (target[:,:,:,4] == 0).unsqueeze(-1).expand_as(target) 
        
        
        
        
        obj_pred = pred[obj_mask].contiguous().view(-1,self.class_num + self.B*5)
        obj_target = target[obj_mask].contiguous().view(-1,self.class_num + self.B*5)
        
        
        no_obj_pred = pred[no_obj_mask].contiguous().view(-1,self.class_num + self.B*5)
        no_obj_target = target[no_obj_mask].contiguous().view(-1,self.class_num + self.B*5)
        no_obj_contain_pred = no_obj_pred[:,4:5*self.B:5].contiguous()
        no_obj_contain_target = no_obj_target[:,4:5*self.B:5].contiguous()

        obj_class_pred = obj_pred[:,5*self.B:]
        obj_class_target = obj_target[:,5*self.B:]
        

        
        
        obj_loc_pred = obj_pred[:,0:5*self.B].contiguous()
        obj_loc_target = obj_target[:,0:5*self.B].contiguous()


        iou = torch.zeros(len(obj_target),self.B)
        iou = Variable(iou)
        
        
        for j in range(self.B):
            pred_bb = torch.zeros(len(obj_target),4)
            pred_bb = Variable(pred_bb)
            
            target_bb = torch.zeros(len(obj_target),4)
            target_bb = Variable(target_bb)
            
            target_bb[:,0] = obj_loc_target[:,j*5] - 0.5*pow(obj_loc_target[:,j*5+2],2)
            target_bb[:,1] = obj_loc_target[:,j*5+1]  - 0.5*pow(obj_loc_target[:,j*5+3],2)
            target_bb[:,2] = obj_loc_target[:,j*5]  + 0.5*pow(obj_loc_target[:,j*5+2],2)
            target_bb[:,3] = obj_loc_target[:,j*5+1]  + 0.5*pow(obj_loc_target[:,j*5+3],2)
            
            pred_bb[:,0] = obj_loc_pred[:,j*5]  - 0.5*pow(obj_loc_pred[:,j*5+2],2)
            pred_bb[:,1] = obj_loc_pred[:,j*5+1]  - 0.5*pow(obj_loc_pred[:,j*5+3],2)
            pred_bb[:,2] = obj_loc_pred[:,j*5]  + 0.5*pow(obj_loc_pred[:,j*5+2],2)
            pred_bb[:,3] = obj_loc_pred[:,j*5+1]  + 0.5*pow(obj_loc_pred[:,j*5+3],2)
            
            iou[:,j] = self.compute_iou(target_bb,pred_bb)
        max_iou,max_index = iou.max(1)
        min_iou,_  = iou.min(1)
        max_index = max_index.data.cpu()

        if if_cuda:
            coo_response_mask = torch.cuda.ByteTensor(obj_target.size(0),self.B*5)
        else:
            coo_response_mask = torch.ByteTensor(obj_target.size(0),self.B*5)
        coo_response_mask.zero_()

        for i in range(len(obj_target)):
            coo_response_mask[i,max_index[i]*5:max_index[i]*5+5] = 1
        
    
        obj_axis_pred = obj_loc_pred[coo_response_mask].contiguous().view(-1,5)
        obj_axis_target = obj_loc_target[coo_response_mask].contiguous().view(-1,5)
        
        
        
        obj_local_loss = F.mse_loss(obj_axis_pred[:,0:2],obj_axis_target[:,0:2],size_average=False) + F.mse_loss(obj_axis_pred[:,2:4],obj_axis_target[:,2:4],size_average=False)
        obj_class_loss = F.mse_loss(obj_class_pred,obj_class_target,size_average=False)
        
        
        if if_cuda:
            max_iou = max_iou.data.double().cuda()
            conf_id = ((1 - max_iou) * self.l_noobj + max_iou ).double().cuda() 
        else:
            max_iou = max_iou.data.double()
            conf_id = ((1 - max_iou) * self.l_noobj + max_iou ).double()
            
        conf_id = Variable(conf_id,requires_grad = True)
        

        obj_contain_loss = F.mse_loss(obj_axis_pred[:,4],max_iou,size_average=False)     
   
        
        no_obj_contain_loss = F.mse_loss(no_obj_contain_pred,no_obj_contain_target,size_average=False)
        


        iou_loss =  F.mse_loss(max_iou,obj_axis_target[:,4],size_average=False)

        loss_all =  (self.l_coord * obj_local_loss + obj_class_loss  + obj_contain_loss + self.l_noobj * no_obj_contain_loss +iou_loss ) / batch_size
        

        loss_info = {
            'local_loss': self.l_coord * obj_local_loss.data,
            'class_loss': obj_class_loss.data,
            'contain_loss': obj_contain_loss.data,
            'no_contain_loss': self.l_noobj * no_obj_contain_loss,
            'iou_loss': iou_loss,
            'mean_iou':torch.mean(max_iou)
        }
        
        return loss_all,loss_info
        
    