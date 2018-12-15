import numpy as np
import torch

import config

def yolo_box_encoder(bs):

    config.YOLO['ceils_size']

    bb_class = np.zeros((config.YOLO['ceils_size'], config.YOLO['ceils_size'], config.YOLO['class_num']))
    bb_response = np.zeros((config.YOLO['ceils_size'], config.YOLO['ceils_size'], config.YOLO['box_num']))
    bb_boxes = np.zeros((config.YOLO['ceils_size'], config.YOLO['ceils_size'], 4 * config.YOLO['box_num']))

    for i in range(bs.shape[0]):

        local_x = int( min(0.99,bs[i,0] + bs[i,2] / 2) * config.YOLO['ceils_size'])
        local_y = int( min(0.99,bs[i,1] + bs[i,3] / 2) * config.YOLO['ceils_size'])

        for j in range(config.YOLO['box_num']):
            bb_response[local_y, local_x, j] = 1

            bb_boxes[local_y, local_x, j * 4 + 0] = bs[i,0] + bs[i,2] /2
            bb_boxes[local_y, local_x, j * 4 + 1] = bs[i,1] + bs[i,3] /2
            bb_boxes[local_y, local_x, j * 4 + 2] = np.sqrt(bs[i,2])
            bb_boxes[local_y, local_x, j * 4 + 3] = np.sqrt(bs[i,3])

        bb_class[local_y, local_x, int(bs[i,4])] = 1

    bb_boxes = torch.from_numpy(bb_boxes).float()
    bb_class = torch.from_numpy(bb_class).float()
    bb_response = torch.from_numpy(bb_response).float()
    boxes = (bb_class,bb_response,bb_boxes)

    return boxes


def yolo_box_decoder(pred,conf_thresh=0.01,nms_thresh=0.5):
    box_scale = config.YOLO['ceils_size']
    cls_num = config.YOLO['class_num']
    box_num = config.YOLO['box_num']
    
    pred_cls, pred_response, pred_bboxes = pred
    
    pred_cls = pred_cls.cpu()
    pred_response = pred_response.cpu()
    pred_bboxes = pred_bboxes.cpu()

    prob = pred_response
    max_prob, max_prob_index = prob.max(3)
    boxes = []
    classes = []
    

    max_prob_index = max_prob_index.permute(1,2,0)
    for B in range(1):
        for cls in range(cls_num):

            cls_prob = (pred_cls[B,:,:,cls]*max_prob[B, :, :]).data
            
            mask_box = torch.zeros((box_scale,box_scale,box_num))
            
            mask_box.scatter_(2,max_prob_index,1)
            mask_box = mask_box.unsqueeze(-1)

            mask_box = mask_box.repeat(1,1,1,4).view(box_scale,box_scale,box_num*4)
            mask_box = mask_box.unsqueeze(0).byte()
            mask_box = mask_box.expand_as(pred_bboxes)
            bbox = pred_bboxes[mask_box].data

            bbox = bbox.reshape(-1,4)
            cls_prob = cls_prob.reshape(-1,1)
            
            a = cls_prob.gt(conf_thresh)
            mask_a = a.expand_as(bbox)

            bbox = bbox[mask_a].reshape(-1,4)
            cls_prob = cls_prob[a].reshape(-1,1)

            if bbox.shape[0] > 0:   
                bbox[:,0:2] = bbox[:,0:2] - 0.5*torch.pow(bbox[:,2:4],2)
                bbox[:,2:4] = torch.pow(bbox[:,2:4],2)
                
                pre_cls_box = bbox.data.numpy()
                pre_cls_score = cls_prob.data.view(-1).numpy()
                
                keep = py_cpu_nms(pre_cls_box, pre_cls_score, thresh=nms_thresh)
                for conf_keep, loc_keep in zip(pre_cls_score[keep], pre_cls_box[keep]):
                    boxes.append(loc_keep)
                    classes.append([cls,conf_keep])

    boxes = np.array(boxes)
    classes = np.array(classes)

    return boxes,classes




def bbox_iou(box1, box2, x1y1x2y2=True):

    mx = np.min((box1[:, 0], box2[:, 0]), axis=0)
    Mx = np.max((box1[:, 2], box2[:, 2]), axis=0)
    my = np.min((box1[:, 1], box2[:, 1]), axis=0)
    My = np.max((box1[:, 3], box2[:, 3]), axis=0)
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1]
    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1]

    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh

    #cw = np.where(cw < 0, 0, cw)
    #ch = np.where(ch < 0, 0, ch)
    cw = np.clip(cw,0,1)
    ch = np.clip(ch,0,1)

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch

    uarea = area1 + area2 - carea

    return carea / uarea


def py_cpu_nms(dets, scores, thresh):
    # dets:(m,5)  thresh:scaler
    #print(scores.shape)
    temp_len = 0  # np.max(dets[:,2]) * 0.05

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = x1 + dets[:, 2]  # dets[:, 2]#
    y2 = y1 + dets[:, 3]  # dets[:, 3]#

    areas = (y2 - y1 + temp_len) * (x2 - x1 + temp_len)

    keep = []

    index = scores.argsort()[::-1][:500]


    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + temp_len)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + temp_len)  # the height of overlap

        overlaps = w * h
        #assert overlaps.all() >= 0
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]

    return keep
