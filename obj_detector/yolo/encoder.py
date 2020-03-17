import numpy as np
import torch

def yolo_encoder(box_list,ceil_size,box_num,cls_num):
    '''
    pred_cls = [C,S,S]
    pred_response = [2,S,S]
    pred_bboxes = [4*2,S,S]
    '''
    box_list.resize((ceil_size,ceil_size))
    labels = box_list.get_field('labels')

    bb_class = np.zeros((cls_num,ceil_size, ceil_size))
    bb_response = np.zeros((box_num,ceil_size, ceil_size))
    bb_boxes = np.zeros((box_num*4,ceil_size, ceil_size))

    #TODO no loop

    for gt,l in zip(box_list.box,labels):


        local_x = int(round((gt[2] - gt[0]) / 2))
        local_y = int(round((gt[3] - gt[1]) / 2))

        for j in range(box_num):
            bb_response[local_y, local_x, j] = 1

            bb_boxes[j * 4 + 0, local_y, local_x] = local_x - gt[0]
            bb_boxes[j * 4 + 1, local_y, local_x] = local_y - gt[1]
            bb_boxes[j * 4 + 2, local_y, local_x] = np.sqrt(gt[2])
            bb_boxes[j * 4 + 3, local_y, local_x] = np.sqrt(gt[3])

        bb_class[l, local_y, local_x] = 1

    bb_boxes = torch.from_numpy(bb_boxes).float()
    bb_class = torch.from_numpy(bb_class).float()
    bb_response = torch.from_numpy(bb_response).float()
    boxes = (bb_class, bb_response, bb_boxes)

    return boxes