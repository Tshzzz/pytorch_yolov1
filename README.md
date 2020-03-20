# [YOLO](https://arxiv.org/abs/1506.02640):  
   
It train YOLO in VOC2007(train,vaild)+VOC2012(train,vaild) datasets and test on VOC2007(test).  
I replace FC layer with Conv Layer to save memory. And you can also training with FC layer!
### Evaluation: 
| Model             | mAp.        |  
| ----------------- | ----------- |  
| My model fc layer   | -         |  
| My model conv layer | 0.64      |  
| Origin papar        | 0.63      |  
  
  
## Dependence:  
- *Python3*  
- *Pytorch 1.3 or higher*  
- *[apex](https://github.com/NVIDIA/apex)*  
  
## Install & Train
### Install Apex  
```  
git clone https://github.com/NVIDIA/apex  
cd apex  
pip install -v --no-cache-dir ./  
```  


### Train on VOCdatasets  
  
1. Download the training, validation, test data and VOCdevkit  
```  
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
 ```  
2. Extract all of these tars into one directory named  `VOCdevkit`  
```  
tar xvf VOCtrainval_06-Nov-2007.tar tar xvf VOCtest_06-Nov-2007.tar tar xvf VOCdevkit_08-Jun-2007.tar
 ```  
3. It should have this basic structure  
```  
 $VOCdevkit/
 $VOCdevkit/VOC2007   
 $VOCdevkit/VOC2012                 
 ```  
4. Generate the train and test list  
```  
python tools/convert_voc.py --dir_path ./ 
cat VOC2007_train.txt VOC2012_train.txt VOC2007_val.txt VOC2012_val.txt >> train.txt
 ```  
  
5. Download the pretrain model  
```
 wget https://pjreddie.com/media/files/darknet19_448.conv.23.
```  
6. Configure the training param   
```
 bash train.sh
```
  
### Test on VOCdatasets  
```
 bash test.sh
```
### Demo 
```
vis_detector.ipynb
```
 
## Samples:  
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/samples/dog.jpg)  
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/samples/person.jpg)  
![imgs](https://github.com/Tshzzz/pytorch_yolov1/raw/master/samples/horses.jpg)

## Reference
[maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
