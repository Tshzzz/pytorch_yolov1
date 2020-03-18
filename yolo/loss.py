
import torch
import torch.nn.functional as F
import torch.nn as nn



class yolov1_loss(nn.Module):
    def __init__(self, B, l_coord, l_noobj, cls_num=20):
        super(yolov1_loss, self).__init__()
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.class_num = cls_num

    def get_kp_torch(self, pred, conf, topk=100):
        b, c, h, w = pred.shape
        pred = pred.contiguous().view(-1)
        pred[pred < conf] = 0
        score, topk_idx = torch.topk(pred, k=topk)

        batch = topk_idx / (h * w * c)

        cls = (topk_idx - batch * h * w * c) / (h * w)

        channel = (topk_idx - batch * h * w * c) - (cls * h * w)

        x = channel % w
        y = channel / w

        return x.view(-1), y.view(-1), cls.view(-1), batch.view(-1)

    def offset2box(self,box,cx,cy):

        box[:, 0] = cx - box[:, 0]
        box[:, 1] = cy - box[:, 1]
        box[:, 2] = box[:, 2] * box[:, 2]
        box[:, 3] = box[:, 3] * box[:, 3]

        """
        cxcywh -> xywh -> xyxy
        """

        box[:, 0] = box[:, 0] - box[:, 2] / 2
        box[:, 1] = box[:, 1] - box[:, 3] / 2
        box[:, 2] = box[:, 0] + box[:, 2]
        box[:, 3] = box[:, 1] + box[:, 3]

        return box

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
        pred_cls, pred_response, pred_bboxes = pred
        label_cls, label_response, label_bboxes = target

        B_size = pred_cls.shape[0]

        device = pred_cls.get_device()
        label_cls = label_cls.to(device)
        label_response = label_response.to(device)
        label_bboxes = label_bboxes.to(device)

        with torch.no_grad():
            tmp_respone =label_response.sum(dim=1).unsqueeze(dim=1)
            k = tmp_respone.sum()
            x_list,y_list,c_list,b_list = self.get_kp_torch(tmp_respone,conf=0.5,topk=int(k))


        t_responses = label_response[b_list,:,y_list,x_list]
        p_responses = pred_response[b_list,:,y_list,x_list]

        t_boxes = label_bboxes[b_list, :, y_list, x_list]
        p_boxes = pred_bboxes[b_list, :, y_list, x_list]

        t_classes = label_cls[b_list, :, y_list, x_list]
        p_classes = pred_cls[b_list, :, y_list, x_list]

        loss_pos_response = 0
        loss_pos_offset = 0
        loss_pos_cls = 0
        for cx,cy,t_offset,p_offset,t_res,p_res,t_cls,p_cls in zip(x_list,y_list,t_boxes,p_boxes,\
                                             t_responses, p_responses, \
                                             t_classes,p_classes
                                        ):
            if t_res.sum() < 0.5:
                break

            t_offset = t_offset.view(-1,4)
            p_offset = p_offset.view(-1,4)

            with torch.no_grad():
                t_box = self.offset2box(t_offset.clone().float(), cx, cy).to(device)
                p_box = self.offset2box(p_offset.clone().float(), cx, cy).to(device)
                iou = self.compute_iou(t_box,p_box)
            #print(iou)
            idx = iou.argmax()
            p_res = p_res[idx]
            loss_pos_response += F.mse_loss(p_res,iou[idx],reduction='sum')
            loss_pos_offset += F.mse_loss(t_offset[idx],p_offset[idx],reduction='sum')
            loss_pos_cls += F.mse_loss(p_cls,t_cls,reduction='sum')


        neg_mask = label_response < 1
        neg_pred = pred_response[neg_mask]
        neg_target = label_response[neg_mask]

        loss_neg_response = F.mse_loss(neg_pred, neg_target, reduction='sum') / B_size
        loss_pos_response = loss_pos_response / B_size
        loss_pos_offset = loss_pos_offset / B_size
        loss_pos_cls = loss_pos_cls / B_size
        loss_obj = loss_neg_response + loss_pos_response


        return {'l_obj': loss_obj, 'l_cls': loss_pos_cls, 'l_offset': loss_pos_offset}


if __name__ == '__main__':
    test_loss = yolov1_loss(2, 5, 0.5)

    batch_size = 7

    label_cls = torch.zeros(batch_size, 20, 7, 7)
    label_bbox = torch.zeros(batch_size,  4 * 2, 7, 7)
    label_response = torch.zeros(batch_size,  2 , 7, 7)


    label_cls[0, 1, 5, 3] = 1
    label_response[0, :, 5, 3] = 1
    label_bbox[0, 0, 5, 3] = 0.1
    label_bbox[0, 1, 5, 3] = 0.1
    label_bbox[0, 2, 5, 3] = 0.2
    label_bbox[0, 3, 5, 3] = 0.3
    label_bbox[0, 4, 5, 3] = 0.1
    label_bbox[0, 5, 5, 3] = 0.1
    label_bbox[0, 6, 5, 3] = 0.2
    label_bbox[0, 7, 5, 3] = 0.3

    pred_cls = torch.zeros(batch_size, 20, 7, 7).to('cuda')
    pred_bbox = torch.zeros(batch_size, 4 * 2, 7, 7).to('cuda')
    pred_response = torch.zeros(batch_size, 2, 7, 7).to('cuda')

    #pred_response[0, 5, 3, :] = 1
    pred_bbox[0, 0, 5, 3] = 0.1
    pred_bbox[0, 1, 5, 3] = 0.1
    pred_bbox[0, 2, 5, 3] = 0.2
    pred_bbox[0, 3, 5, 3] = 0.3
    pred_bbox[0, 4, 5, 3] = 0.1
    pred_bbox[0, 5, 5, 3] = 0.1
    pred_bbox[0, 6, 5, 3] = 0.2
    pred_bbox[0, 7, 5, 3] = 0.3


    pred = (pred_cls, pred_response, pred_bbox)
    target = (label_cls, label_response, label_bbox)
    test_loss(pred,target)



