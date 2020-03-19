import torch
import numpy as np

from structure.bounding_box import BoxList
from torchvision.ops import nms
from collections import defaultdict

def get_top_torch(pred, conf, topk=100):
    c, h, w = pred.shape
    pred = pred.view(-1)
    pred[pred < conf] = 0
    topk = min(len(pred), topk)
    score, topk_idx = torch.topk(pred, k=topk)
    ceil = (topk_idx / (h * w))
    channel = topk_idx - ceil * h * w
    x = channel % w
    y = channel / w
    return x.view(-1), y.view(-1), ceil.view(-1)



def yolo_decoder(pred, img_size, conf=0.02, topk=100, nms_threshold=0.5):
    '''
    pred_cls = [C,S,S]
    pred_response = [2,S,S]
    pred_bboxes = [4*2,S,S]
    '''

    pred_cls, pred_response, pred_bboxes = pred
    class_num, h, w = pred_cls.shape

    x_list, y_list, b_list = get_top_torch(pred_response, conf=conf, topk=topk)

    collect_box = defaultdict(list)
    collect_score = defaultdict(list)

    for i in range(len(b_list)):

        box_idx = b_list[i]
        x = x_list[i]
        y = y_list[i]
        label = pred_cls[:, y, x].argmax()
        score = pred_cls[label, y, x] * pred_response[box_idx,y,x]
        if score < conf:
            break
        select_box = (box_idx )*4
        offsets = pred_bboxes[select_box:select_box+4, y, x]
        ox, oy, bw, bh = offsets
        cx = x - ox
        cy = y - oy
        bw = bw * bw
        bh = bh * bh
        bw *= w
        bh *= h


        x1 = torch.clamp(cx - bw / 2, 0, w).unsqueeze(dim=0)
        y1 = torch.clamp(cy - bh / 2, 0, h).unsqueeze(dim=0)
        x2 = torch.clamp(cx + bw / 2, 0, w).unsqueeze(dim=0)
        y2 = torch.clamp(cy + bh / 2, 0, h).unsqueeze(dim=0)

        bbox = torch.cat((x1, y1, x2, y2), dim=0)
        collect_box[int(label)].append(bbox.tolist())
        collect_score[int(label)].append(score.tolist())

    bboxes = []
    scores = []
    labels = []
    keys = list(collect_box.keys())
    for k in keys:
        if len(collect_box[k]) > 0:
            bbox = np.asarray(collect_box[k])
            score = np.asarray(collect_score[k])
            bbox = torch.from_numpy(bbox).float()
            score = torch.from_numpy(score).float()
            keep = nms(bbox, score,nms_threshold)
            scores += score[keep].tolist()
            labels += [k for _ in range(len(keep))]
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

    box.resize(img_size)

    return box

def yolo_decoder_old(pred, img_size, conf=0.02, topk=100, nms_threshold=0.5):
    '''
    pred_cls = [S,S,C]
    pred_response = [S,S,2]
    pred_bboxes = [S,S,2*4]
    '''

    pred_cls, pred_response, pred_bboxes = pred
    pred_cls = pred_cls.permute(2,0,1).contiguous()
    pred_bboxes = pred_bboxes.permute(2,0,1).contiguous()
    pred_response = pred_response.permute(2,0,1).contiguous()



    class_num, h, w = pred_cls.shape

    x_list, y_list, b_list = get_top_torch(pred_response, conf=conf, topk=topk)


    collect_box = defaultdict(list)
    collect_score = defaultdict(list)

    for i in range(len(b_list)):

        box_idx = b_list[i]
        x = x_list[i]
        y = y_list[i]
        label = pred_cls[:, y, x].argmax()
        score = pred_cls[label, y, x] * pred_response[box_idx,y,x]
        if score < conf:
            break
        select_box = (box_idx )*4
        offsets = pred_bboxes[select_box:select_box+4, y, x]

        cx, cy, bw, bh = offsets
        bw = bw * bw
        bh = bh * bh
        bw *= w
        bh *= h

        x1 = torch.clamp(cx - bw / 2, 0, w).unsqueeze(dim=0)
        y1 = torch.clamp(cy - bh / 2, 0, h).unsqueeze(dim=0)
        x2 = torch.clamp(cx + bw / 2, 0, w).unsqueeze(dim=0)
        y2 = torch.clamp(cy + bh / 2, 0, h).unsqueeze(dim=0)

        bbox = torch.cat((x1, y1, x2, y2), dim=0)
        collect_box[int(label)].append(bbox.tolist())
        collect_score[int(label)].append(score.tolist())

    bboxes = []
    scores = []
    labels = []
    keys = list(collect_box.keys())
    for k in keys:
        if len(collect_box[k]) > 0:
            bbox = np.asarray(collect_box[k])
            score = np.asarray(collect_score[k])
            bbox = torch.from_numpy(bbox).float()
            score = torch.from_numpy(score).float()
            keep = nms(bbox, score,nms_threshold)
            scores += score[keep].tolist()
            labels += [k for _ in range(len(keep))]
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

    box.resize(img_size)

    return box