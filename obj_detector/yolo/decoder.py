import torch
import numpy as np

from structure.bounding_box import BoxList
from torchvision.ops import nms

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

    x_list, y_list, c_list = get_top_torch(pred_cls, conf=conf, topk=topk)
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
            box_idx = pred_response[:, y, x].argmax()

            score = pred_response[box_idx, y, x] * cls_score

            mask_score = score > conf

            if mask_score.sum() <= 0:
                continue

            score = score[mask_score]
            x = x[mask_score]
            y = y[mask_score]

            offsets = pred_bboxes[box_idx:box_idx + 4, :, :]

            ox, oy, bw, bh = offsets

            # ceil center
            cx = (x + 0.5) + ox
            cy = (y + 0.5) + oy
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
    box.resize(img_size)
    return box