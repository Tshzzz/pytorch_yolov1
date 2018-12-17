classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


bbox_num = 2
box_scale = 7
lr = 1e-4


train_image_list = "./train.txt"
eval_image_list= "./VOC2007_test.txt"
model_path = "./runs/"
pretrain_path = '../pytorch_yolov1/darknet19_448.conv.23'

VOC_root_dir = '/home/tshzzz/Imagesets/VOCdevkit'


model_save_iter = 1  # epochs
epochs_start = 140
epochs_end = 150
batch_size = 20
cls_num = len(classes)

use_conv = False

## data augmentation

jitter = 0.2
hue = 0.1
saturation = 1.5
exposure = 1.5

#loss
l_coord = 5
l_noobj = 0.5





YOLO = {
    'ceils_size':7,
    'class_num':len(classes),
    'box_num':2,
    'image_size':[448,448]
}



