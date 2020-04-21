import torch
from tqdm import tqdm
from metric import base_val
from data.datasets import VOCDatasets
from data.evaluate.voc_eval import voc_evaluation
from yolo.encoder import yolo_encoder
from yolo.decoder import yolo_decoder




def test_encoder_decoder():

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    test_file = './VOC2007_test.txt'
    load = VOCDatasets(test_file)

    img_info = dict()
    gt_info = dict()
    predictions = []
    print("eval {} total img {}".format(len(classes),len(load)))

    for meta in tqdm(load):

        fileID = meta['fileID']
        img_info[fileID] = dict(width=meta['img_width'],height=meta['img_height'])
        gt_info[fileID] = meta['boxlist']
        scale_gt = meta['boxlist'].copy()

        cls,obj,box = yolo_encoder(scale_gt,(7,7),2,len(classes))
        target_cls = torch.from_numpy(cls).float()
        target_obj = torch.from_numpy(obj).float()
        target_box = torch.from_numpy(box).float()
        target = (target_cls,target_obj,target_box)
        box = yolo_decoder(target,(meta['img_width'],meta['img_height']))

        predictions.append([fileID,box])

    gt_sets = base_val(img_info,gt_info,classes)

    result = voc_evaluation(gt_sets,predictions,'./',box_only=True)
    for i,value in enumerate(result['ap']):
        result[classes[i]] = value
    del result['ap']

    return


if __name__ == "__main__":
    test_encoder_decoder()

