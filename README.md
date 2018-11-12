## Yolo v1: 
It train YOLO in VOC2007(train,vaild)+VOC2012(train,vaild) datasets and test on VOC2007(test).

I  change the last two fc layer into conv layer. 

In the conv prediction model, the predicition speed is faster and model weight is smaller.

### Accuracy:
| Model             | mAp.        |
| ----------------- | ----------- |
| My model fc layer   | 0.59      |
| My model conv layer | 0.54      |
| Origin papar        | 0.63      |


### Conclusions:
Data Augmentation is crucial ! The random crop help me improve more than 10% mAp.


### Samples:
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/samples/dog.jpg)
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/samples/person.jpg)
