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
            tmp_boxes = gt_list.box
            if tmp_boxes.shape[0] >= 3:

                tmp_labels = gt_list.get_field('labels')
                rm_idx = random.randint(0, gt_list.box.shape[0] - 1)
                fill_bk = tmp_boxes[rm_idx]
                boxes = np.delete(tmp_boxes, rm_idx, axis=0)

                labels = tmp_labels[:rm_idx] + tmp_labels[rm_idx + 1:]
                max_iou = get_max_overlap(fill_bk,boxes)

                if max_iou < 0.3:
                    gt_list.box = boxes
                    gt_list.add_field('labels',labels)
                    fill_bk = fill_bk.astype(np.int)
                    w = int(fill_bk[2] - fill_bk[0])
                    h = int(fill_bk[3] - fill_bk[1])
                    noise = np.random.normal(0, 255, (h, w, 3))
                    img[fill_bk[1]:fill_bk[3], fill_bk[0]:fill_bk[2], :] = noise


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


