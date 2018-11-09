classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#
bbox_num = 2
box_scale = 7
lr = 1e-5


train_image_list = "./train_list/train.txt"
eval_image_list= "./train_list/VOC2007_test.txt"
model_path = "./runs/"



model_save_iter = 1  # epochs
epochs_start = 50
epochs_end = 100
batch_size = 30
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





