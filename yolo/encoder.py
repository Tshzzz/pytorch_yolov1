import numpy as np

def yolo_encoder(box_list,ceil_size,box_num,cls_num):
    '''
    pred_cls = [C,S,S]
    pred_response = [2,S,S]
    pred_bboxes = [4*2,S,S]
    '''
    w,h = ceil_size
    box_list.resize(ceil_size)
    labels = box_list.get_field('labels')

    bb_class = np.zeros((cls_num,h, w))
    bb_response = np.zeros((box_num,h, w))
    bb_boxes = np.zeros((box_num*4,h, w))

    #TODO avoid loop
    for gt,l in zip(box_list.box,labels):
        local_x = min(int(round((gt[2] + gt[0]) / 2)),w-1)
        local_y = min(int(round((gt[3] + gt[1]) / 2)),h-1)

        for j in range(box_num):
            bb_response[j, local_y, local_x] = 1
            bb_boxes[j * 4 + 0, local_y, local_x] = (gt[2] + gt[0])/2
            bb_boxes[j * 4 + 1, local_y, local_x] = (gt[3] + gt[1])/2
            bb_boxes[j * 4 + 2, local_y, local_x] = max((gt[2] - gt[0]),0.01)
            bb_boxes[j * 4 + 3, local_y, local_x] = max((gt[3] - gt[1]),0.01)

        bb_class[l, local_y, local_x] = 1
    boxes = (bb_class, bb_response, bb_boxes)
    return boxes

