train_cfg = dict()
train_cfg['lr'] = [0.5e-3, 1e-5]
train_cfg['epochs'] = 75
train_cfg['milestone'] = [40,60]
train_cfg['gamma'] = 0.1
train_cfg['batch_size'] = 1
train_cfg['gpu_id'] = [0,1]
train_cfg['out_dir'] = 'experiment/VOCNet'
train_cfg['resume'] = False
train_cfg['use_sgd'] = True
train_cfg['device'] = 'cuda'

train_cfg['dataroot'] = './'

train_cfg['img_size'] = [448]

train_cfg['classes'] = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]



model_cfg = dict()
model_cfg['model_type'] = 'DarkNet'
model_cfg['class_num'] = len(train_cfg['classes'])
model_cfg['backbone'] = 19
model_cfg['box_num'] = 2
model_cfg['ceil_size'] = 7
model_cfg['pretrained'] = None#'darknet19_448.conv.23'
model_cfg['l_coord'] = 3
model_cfg['l_obj'] = 3
model_cfg['l_noobj'] = 0.5
model_cfg['conv_mode'] = True

cfg = dict()

cfg['train_cfg'] = train_cfg
cfg['model_cfg'] = model_cfg

