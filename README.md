## Yolo v1: 
It's too hard to train yolo v1. I'm still can't recurrent the accuracy in voc datasets. 

I train on voc2012 trainsets , the mAp on trainsets can reach to 80. but on the testsets it only has 30. 

I train a single class detect net in coco datasets , it works but still not well. 

I think the yolo has too few boxes to predict the object , and hard to learn the matching strategies.

### Train Method:
I pick the images with person class from coco datasets . The trainsets contains about 11k picture.

I train 100 epochs with lr 0.0001.

You can reset the train parameter to train your datasets.

the training logs:
![imgs](https://raw.githubusercontent.com/Tshzzz/pytorch_yolov1/master/imgs/train_log.png)


### Results:
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/imgs/000000001591.jpg)
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/imgs/000000000692.jpg)
