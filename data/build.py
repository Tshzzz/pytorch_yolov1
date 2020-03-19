from data.datasets import *
from torch.utils import data
import torch
from  torch.utils.data.distributed import DistributedSampler
import cv2
import random
import numpy as np
import math
from yolo.encoder import yolo_encoder_old

class MutilScaleBatchCollator(object):
    def __init__(self,img_size,train,encoder=None):
        self.img_size = img_size
        self.train = train
        self.encoder = yolo_encoder_old#encoder
    # TODO padding img to the same size
    # TODO Training with original size
    def covert_img_tensor(self,img):
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img[0, :, :] = (img[0, :, :] - 0.485) / 1
        img[0, :, :] = (img[0, :, :] - 0.456) / 1
        img[0, :, :] = (img[0, :, :] - 0.406) / 1

        return img

    def process_mask(self,meta,img_size):
        images = []
        targets = []

        if False:#self.train and random.random() > 0.5:
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
                padding_img = torch.from_numpy(padding_img).permute(2,0,1).float()
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

        if self.encoder:
            target_cls = []
            target_obj = []
            target_box = []
            for t in targets:
                cls,obj,box = self.encoder(t)
                target_cls.append(cls)
                target_obj.append(obj)
                target_box.append(box)

            target_cls = torch.from_numpy(np.array(target_cls)).float()
            target_obj = torch.from_numpy(np.array(target_obj)).float()
            target_box = torch.from_numpy(np.array(target_box)).float()
            targets = [target_cls,target_obj,target_box]


        return batch_imgs,targets,meta


def make_mutilscale_voc_loader(list_path,train=False,img_size=[(512,512)],batch_size=4,num_workers=4):

    dataset = VOCDatasets(list_path,train)
    collator = MutilScaleBatchCollator(img_size,train)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  num_workers=num_workers,
                                  collate_fn=collator,
                                  pin_memory=True
                                  )
    return data_loader




def make_dist_voc_loader(list_path,train=False,img_size=[(512,512)],
                         batch_size=4,num_workers=4,num_replicas=2,rank=0):


    dataset = VOCDatasets(list_path,train)
    collator = MutilScaleBatchCollator(img_size,train)
    sampler =DistributedSampler(dataset,num_replicas=num_replicas,rank=rank,shuffle=train)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collator,
                                  sampler=sampler,
                                  pin_memory=True
                                  )

    return data_loader


