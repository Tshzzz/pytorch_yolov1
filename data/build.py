from data.datasets import *
from torch.utils import data
import torch
from  data.sampler import TrainingSampler,InferenceSampler
import cv2
import random
import numpy as np
import math
from yolo.encoder import yolo_encoder


class MutilScaleBatchCollator(object):
    def __init__(self,img_size,train,encoder_param):
        self.img_size = img_size
        self.train = train
        self.encoder = yolo_encoder
        if encoder_param:
            self.cls_num = encoder_param['class_num']
            self.box_num = encoder_param['box_num']
            self.ceil_size = encoder_param['ceil_size']

    def process_mask(self,meta,img_size):
        images = []
        targets = []

        if self.train and random.random() > 0.5:
            max_size = max([max(info['img_width'],info['img_height']) for info in meta])
            max_size = math.ceil(max_size / 32) * 32
            for info in meta:
                img = info['img']
                h,w,c = img.shape
                padding_img = np.zeros((max_size,max_size,c),dtype=np.uint8)
                padding_img[:h,:w,:c] = img
                gt_list = info['boxlist'].copy()
                gt_list.size = (max_size,max_size)

                padding_img = cv2.resize(padding_img, img_size)
                padding_img = torch.from_numpy(padding_img).permute(2,0,1).float() /255.
                images.append(padding_img)
                gt_list.resize(img_size)
                targets.append(gt_list)
        else:
            for info in meta:
                img = info['img']
                img = cv2.resize(img, img_size)
                img = torch.from_numpy(img).permute(2, 0, 1).float() /255.
                images.append(img)
                gt_list = info['boxlist'].copy()
                gt_list.resize(img_size)
                targets.append(gt_list)
        return images,targets

    def __call__(self, batch):
        meta = list(batch)
        sized = random.choice(self.img_size)
        images,targets = self.process_mask(meta,img_size=sized)
        batch_imgs = torch.cat([a.unsqueeze(0) for a in images])

        if self.train and self.encoder:
            target_cls = []
            target_obj = []
            target_box = []
            for t in targets:
                cls,obj,box = self.encoder(t,self.ceil_size,self.box_num,self.cls_num)
                target_cls.append(cls)
                target_obj.append(obj)
                target_box.append(box)

            target_cls = torch.from_numpy(np.array(target_cls)).float()
            target_obj = torch.from_numpy(np.array(target_obj)).float()
            target_box = torch.from_numpy(np.array(target_box)).float()
            targets = [target_cls,target_obj,target_box]


        return batch_imgs,targets,meta




def make_dist_voc_loader(list_path,encoder_param=None,train=False,img_size=[(448,448)],
                         batch_size=4,num_workers=4):


    dataset = VOCDatasets(list_path,train)
    collator = MutilScaleBatchCollator(img_size,train,encoder_param)
    if train:
        sampler =TrainingSampler(len(dataset),shuffle=train)
    else:
        sampler = InferenceSampler(len(dataset))

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collator,
                                  sampler=sampler,
                                  pin_memory=True
                                  )

    return data_loader


