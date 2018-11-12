## Yolo v1: 
It train YOLO in VOC2007(train,vaild)+VOC2012(train,vaild) datasets and test on VOC2007(test).
And I  change the last two fc layer into conv layer. In the conv prediction model, the predicition speed
is faster and model weight is smaller.
| Model             | mAp.        |
|-------------------| ------------|
| My model fc layer   | 0.59      |
| My model conv layer | 0.54      |
| Origin papar         | 0.63      |

### Conclusions:
Data Augmentation is crucial ! The random crop help me improve more than 10% mAp.

### Train on VOCdatasets
1.  Download the training, validation, test data and VOCdevkit
```    
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```    
2.  Extract all of these tars into one directory named  `VOCdevkit`
```    
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
```    
3.  It should have this basic structure
```    
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
```    
4.  Generate the train and test list
```
	cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt
	python convert_voc.py --dir_path ./
```

5.  Download the pretrain model
	You can download pretrain model in (https://pjreddie.com/media/files/darknet19_448.conv.23) .
	
6.  Configure the training param 
	you can see the config.py, and change the right paths while training your datasets.
```
	python train.py
```
### Samples:
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/samples/dog.jpg)
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/samples/person.jpg)
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/samples/person.jpg)
