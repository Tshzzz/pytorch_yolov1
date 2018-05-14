## Yolo v1: 
	It's too hard to train yolo v1. I'm still can't recurrent the accuracy in voc datasets. 
	I train a person class detect net in coco datasets , it works but still not well. 
	I think the yolo has too little boxes to predict the object.

### Train Method:
	I pick the images with person class from coco datasets . The trainsets contains about 11k picture.
	I train 100 epochs with lr 0.0001.
	You can reset the train parameter to train your datasets.
	git push -u origin master
	![imgs](https://github.com/Tshzzz/pytorch_yolov1/imgs/train_log.png)
	the training logs.

### Highlights:
	![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/imgs/000000001319.jpg)
	![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/imgs/000000001237.jpg)
	![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/imgs/000000001591.jpg)
	![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/imgs/000000000692.jpg)
