
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


    def loss_cls(self,pred,target):

        pos_mask = target > 0

        pos_pred = pred[pos_mask]
        pos_target = target[pos_mask]
        neg_pred = pred[pos_mask==False]
        neg_target = target[pos_mask==False]

        loss_pos = F.mse_loss(pos_pred, pos_target, reduction='sum') * 3
        loss_neg = F.mse_loss(neg_pred, neg_target, reduction='sum')
        loss =  loss_neg + loss_pos

        return loss

    """
    pred = (b,2*4,7,7)
    target = (b,2*4,7,7)
    """
    def loss_offsets(self,pred,target):
        pos_mask = target > 0

        pos_pred = pred[pos_mask]#.view(-1,4)
        pos_target = target[pos_mask]#.view(-1,4)
        neg_pred = pred[pos_mask==False]
        neg_target = target[pos_mask==False]

        loss_pos = F.mse_loss(pos_pred, pos_target, reduction='sum') * 3
        loss_neg = F.mse_loss(neg_pred, neg_target, reduction='sum')
        loss =  loss_neg + loss_pos

        return loss

    """
    pred = (b,2,7,7)
    target = (b,2,7,7)
    """
    def loss_response(self,pred,target):

        pos_mask = target > 0



        pos_pred = pred[pos_mask]
        pos_target = target[pos_mask]
        neg_pred = pred[pos_mask==False]
        neg_target = target[pos_mask==False]

        loss_pos = F.mse_loss(pos_pred, pos_target, reduction='sum') * 3
        loss_neg = F.mse_loss(neg_pred, neg_target, reduction='sum')

        loss =  loss_neg + loss_pos

        return loss

    def forward(self,pred,target):
        pred_cls, pred_response, pred_bboxes = pred
        label_cls, label_response, label_bboxes = target
        device = pred_cls.get_device()
        label_cls = label_cls.to(device)
        label_response = label_response.to(device)
        label_bboxes = label_bboxes.to(device)

        loss_obj = self.loss_response(pred_response, label_response)
        loss_cls = self.loss_cls(pred_cls, label_cls)
        loss_offset = self.loss_offsets(pred_bboxes, label_bboxes)
        #loss_all  = loss_obj + loss_cls + loss_offset

        return {'l_obj': loss_obj, 'l_cls': loss_cls, 'l_offset': loss_offset}

if __name__ == '__main__':
    test_loss = yolov1_loss(2, 5, 0.5)

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

    #pred_response[0, 5, 3, :] = 1
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



