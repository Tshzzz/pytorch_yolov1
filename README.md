
### TODO
- [x] CenterNet
- [ ] Data Augment
- [ ] YOLO v2,v3
- [ ] FCOS  
- [ ] Retinanet



### Experiment
#### Center Net
| Backbone          | data method | img size    |    mAp      |
| ------------------| ----------- | ------------| ----------- |
| ResNet34          |  none       | 384 x 384   |    0.66     |
| ResNet34          |  vflip      | 384 x 384   |    0.70     |
| ResNet34          | ramdon crop | 384 x 384   |    0.728    |
| ResNet34          |  erase box  | 384 x 384   |    0.742    |
| ResNet50          |  ALL        | 384 x 384   |    0.751    |
| ResNet50          |  ALL        | 384 x 384   |    0.775    |
| ResNet101         |  ALL        | 384 x 384   |    0.759    |
