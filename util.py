import torch
import config

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
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

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    #print(boxes)
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes

def decoder_(pred_cls,pred_response,pred_bboxes):
    prob = pred_response
    max_prob,max_prob_index = prob.max(3)
    boxes=[]
    for k in range(1):
        for i in range(config.box_scale):
            for j in range(config.box_scale):  
                cls_prob,cls = pred_cls[k,i,j,:].max(0)
                if max_prob[k,i,j].data.numpy()* cls_prob.data  > 0.1:
                    max_prob_index_np = max_prob_index[k,i,j].data.numpy()

                    bbox = pred_bboxes[k , i , j , max_prob_index_np*4 : max_prob_index_np*4 + 4].contiguous().data   
                    
                    box_xy = torch.zeros(6)
                    box_xy[:2] = bbox[:2] - 0.5*pow(bbox[2:4],2)
                    box_xy[2:4] = bbox[:2] + 0.5*pow(bbox[2:4],2)
                    
                    box_xy[4] = max_prob[k,i,j].data * cls_prob.data
                    box_xy[5] = int(cls.data.numpy())
                    boxes.append(box_xy.view(1,6).numpy()[0].tolist())
                    
    boxes = nms(boxes, 0.5)
    return boxes

def decoder_vaild(pred_cls,pred_response,pred_bboxes,cls_num):    
    prob = pred_response
    
    max_prob,max_prob_index = prob.max(3)
    boxes=[]

    for k in range(1):
        for i in range(config.box_scale):
            for j in range(config.box_scale):  
                cls_prob,cls = pred_cls[k,i,j,:].max(0)
                if max_prob[k,i,j].data.numpy()* cls_prob.data  > 0.1:
                    max_prob_index_np = max_prob_index[k,i,j].data.numpy()

                    bbox = pred_bboxes[k , i , j , max_prob_index_np*4 : max_prob_index_np*4 + 4].contiguous().data   
                    
                    
                    box_xy = torch.zeros(cls_num+5)
                    box_xy[:2] = bbox[:2] - 0.5*pow(bbox[2:4],2)
                    box_xy[2:4] = bbox[:2] + 0.5*pow(bbox[2:4],2)
                    box_xy[4] = max_prob[k,i,j] #* cls_prob.data
                    box_xy[5:] = pred_cls[k,i,j,:]
                    boxes.append(box_xy.view(1,cls_num+5).data.numpy()[0].tolist())
                    
    boxes = nms(boxes, 0.5)
    return boxes