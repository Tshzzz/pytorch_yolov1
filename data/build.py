from data.datasets import *
from torch.utils import data
import torch
from  data.sampler import TrainingSampler,InferenceSampler
import cv2
import random
import numpy as np
import math


class MutilScaleBatchCollator(object):
    def __init__(self,img_size,train):
        self.img_size = [a for a in range(min(img_size),max(img_size)+32,32)]
        #print(self.img_size)
        self.train = train

    def normlize(self,img):

        img = np.float32(img) if img.dtype != np.float32 else img.copy()

        return img/255.

    def process_image(self,meta,sized):
        images = []


        for info in meta:
            img = info['img']
            padding_img = img.copy()
            info['padding_width'] = padding_img.shape[1]
            info['padding_height'] = padding_img.shape[0]
            img_size = [sized, sized]

            img_size[0] = math.ceil(img_size[0] / 32) * 32
            img_size[1] = math.ceil(img_size[1] / 32) * 32
            img_size = (img_size[0],img_size[1])

            padding_img = cv2.resize(padding_img, img_size)
            padding_img = self.normlize(padding_img)
            padding_img = torch.from_numpy(padding_img).permute(2,0,1).float()
            images.append(padding_img)

        return images

    def __call__(self, batch):
        meta = list(batch)
        if self.train:
            sized = random.choice(self.img_size)
        else:
            sized = sum(self.img_size) / float(len(self.img_size))

        images = self.process_image(meta,sized)
        batch_imgs = torch.cat([a.unsqueeze(0) for a in images])

        return batch_imgs,meta



def make_dist_voc_loader(list_path,train=False,img_size=[(448,448)],
                         batch_size=4,num_workers=4):


    dataset = VOCDatasets(list_path,train)
    collator = MutilScaleBatchCollator(img_size,train)
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


