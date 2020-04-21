import numpy as np
from structure.bounding_box import BoxList
from torchvision.ops import nms


def yolo_decoder(pred, img_size, conf=0.02, nms_threshold=0.5):
    '''
    pred_cls = [C*2,S,S]
    pred_response = [2,S,S]
    pred_bboxes = [4*2,S,S]
    '''

    pred_cls, pred_response, pred_bboxes = pred
    class_num, h, w = pred_cls.shape
    box_num = pred_response.shape[0]

    pred_cls = pred_cls.view(1,class_num, h,w).permute(2,3,0,1).contiguous()
    pred_cls = pred_cls.repeat(1,1,box_num,1).view(-1, class_num)

    pred_bboxes = pred_bboxes.view(box_num,4, h, w).permute(2,3,0,1).contiguous().view(-1, 4)
    pred_response = pred_response.view(box_num,1, h, w).permute(2,3,0,1).contiguous().view(-1, 1)

    # 找最anchor中置信度最高的
    pred_mask = (pred_response > conf).view(-1)
    pred_bboxes = pred_bboxes[pred_mask]
    pred_response = pred_response[pred_mask]
    pred_cls = pred_cls[pred_mask]

    bboxes = []
    scores = []
    labels = []

    for cls in range(class_num):
        score = pred_cls[:,cls].float() * pred_response[:, 0]
        mask_a = score.gt(conf)
        bbox = pred_bboxes[mask_a]
        cls_prob = score[mask_a]
        if bbox.shape[0] > 0:
            bbox[:, 2] = bbox[:, 2] * bbox[:, 2]
            bbox[:, 3] = bbox[:, 3] * bbox[:, 3]

            bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
            bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
            bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
            bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
            pre_cls_box = bbox.data
            pre_cls_score = cls_prob.data.view(-1)

            keep = nms(pre_cls_box, pre_cls_score, nms_threshold)

            for conf_keep, loc_keep in zip(pre_cls_score[keep], pre_cls_box[keep]):
                bboxes.append(loc_keep.tolist())
                scores.append(conf_keep.tolist())
                labels.append(cls)



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