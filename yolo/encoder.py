import numpy as np

def yolo_encoder(box_list,ceil_size=7,box_num=2,cls_num=20):
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

    #TODO avoid loop
    for gt,l in zip(box_list.box,labels):
        local_x = min(int(round((gt[2] + gt[0]) / 2)),ceil_size-1)
        local_y = min(int(round((gt[3] + gt[1]) / 2)),ceil_size-1)

        for j in range(box_num):
            bb_response[j, local_y, local_x] = 1
            bb_boxes[j * 4 + 0, local_y, local_x] = (local_x - (gt[2] + gt[0])/2)
            bb_boxes[j * 4 + 1, local_y, local_x] = (local_y - (gt[3] + gt[1])/2)
            bb_boxes[j * 4 + 2, local_y, local_x] = np.sqrt(max((gt[2] - gt[0])/ceil_size,0.01))
            bb_boxes[j * 4 + 3, local_y, local_x] = np.sqrt(max((gt[3] - gt[1])/ceil_size,0.01))

        bb_class[l, local_y, local_x] = 1
    boxes = (bb_class, bb_response, bb_boxes)
    return boxes

def yolo_encoder_old(box_list,ceil_size=7,box_num=2,cls_num=20):
    '''
    pred_cls = [S,S,C]
    pred_response = [S,S,2]
    pred_bboxes = [S,S,4*2]
    '''
    box_list.resize((ceil_size,ceil_size))
    labels = box_list.get_field('labels')

    bb_class = np.zeros((ceil_size, ceil_size,cls_num))
    bb_response = np.zeros((ceil_size, ceil_size,box_num))
    bb_boxes = np.zeros((ceil_size, ceil_size,box_num*4))

    #TODO avoid loop
    for gt,l in zip(box_list.box,labels):
        local_x = min(int(round((gt[2] + gt[0]) / 2)),ceil_size-1)
        local_y = min(int(round((gt[3] + gt[1]) / 2)),ceil_size-1)

        for j in range(box_num):
            bb_response[local_y, local_x, j] = 1

            bb_boxes[local_y, local_x,j * 4 + 0] = (gt[2] + gt[0]) / 2
            bb_boxes[local_y, local_x,j * 4 + 1] = (gt[3] + gt[1]) / 2
            bb_boxes[local_y, local_x,j * 4 + 2] = np.sqrt(max((gt[2] - gt[0])/ceil_size,0.01))
            bb_boxes[local_y, local_x,j * 4 + 3] = np.sqrt(max((gt[3] - gt[1])/ceil_size,0.01))

        bb_class[local_y, local_x,l] = 1
    boxes = (bb_class, bb_response, bb_boxes)
    return boxes
