from torch.utils import data
from structure.bounding_box import *
import cv2
import random
from data.datasets._utils import get_random_crop_tran,get_max_overlap,random_affine,Grid
import torch

class VOCDatasets(data.Dataset):
    def __init__(self, list_file,train=False):

        self.train = train
        self.label_path = []
        self.image_path = []

        with open(list_file) as f:
            lines = f.readlines()
        self.num_samples = len(lines)
        for line in lines:
            splited = line.strip().split(' ')
            self.image_path.append(splited[0])
            self.label_path.append(splited[1])

        self.grid = Grid(True, True, rotate=1,offset= 0,ratio= 0.5, mode=1, prob=0.7)

    def _get_label(self, file, size):
        tmp = open(file, 'r')
        gt = []
        labels = []
        difficult = []
        for f in tmp.readlines():
            a = list(map(float, f.strip().split(',')))
            gt.append(a[0:4])
            labels.append(int(a[4]))
            difficult.append(0)
        tmp.close()
        gt_list = BoxList(gt, size)
        gt_list.add_field('labels', labels)
        gt_list.add_field('difficult', np.asarray(difficult))
        return gt_list

    def _get_img(self, img_file):
        img = cv2.imread(img_file)[:, :, ::-1].copy()
        return img

    def get_data(self, idx):
        file_name = self.image_path[idx]
        gt_path = self.label_path[idx]
        img = self._get_img(file_name)
        gt_list = self._get_label(gt_path, (img.shape[1], img.shape[0]))

        if self.train:
            img,gt_list = self._data_aug(img,gt_list)
            img = img.copy()



        meta = dict()
        meta['fileID'] = gt_path.split('.')[0].split('/')[-1].replace('.txt', '')
        meta['img_width'] = img.shape[1]
        meta['img_height'] = img.shape[0]
        meta['boxlist'] = gt_list.copy()
        meta['img'] = img

        return meta


    def _data_aug(self,img,gt_list):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            gt_list.flip(1)

        if random.random() > 0.5:
            img = torch.from_numpy(img) / 255.
            img = img.permute((2, 0, 1))

            img, label = self.grid(img, gt_list)
            img = img.permute((1, 2, 0))
            img = img * 255
            img = img.numpy()
            img = img.astype(np.uint8)

        if random.random() > 0.2:
            img,gt_list = random_affine(img,gt_list,degrees=5, translate=.1, scale=.1, shear=2, border=0)


        if random.random() > 0.2:

            matrix = get_random_crop_tran(img)
            h, w, _ = img.shape
            img = cv2.warpAffine(img, matrix, (w, h))
            gt_list.warpAffine(matrix, (w, h))



        return img,gt_list


    def __getitem__(self, idx):
        meta = self.get_data(idx)
        return meta

    def __len__(self):
        return self.num_samples


