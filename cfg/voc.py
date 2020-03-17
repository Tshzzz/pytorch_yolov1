train_cfg = dict()
train_cfg['lr'] = [1e-2, 1e-5]
train_cfg['epochs'] = 75
train_cfg['milestone'] = [10,20,40,60]
train_cfg['gamma'] = 0.5
train_cfg['batch_size'] = 1
train_cfg['gpu_id'] = [0,1]
train_cfg['scale'] = [8,16,32,64,128]
train_cfg['out_dir'] = 'experiment/VOCNet'
train_cfg['resume'] = False
train_cfg['use_sgd'] = True
train_cfg['device'] = 'cuda'

train_cfg['dataroot'] = './'

train_cfg['patch_size'] = [(448,448)]

train_cfg['classes'] = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]



model_cfg = dict()
model_cfg['model_type'] = 'DarkNet'
model_cfg['class_num'] = len(train_cfg['classes'])
model_cfg['backbone'] = 19
model_cfg['box_num'] = 2
model_cfg['ceil_size'] = 7
model_cfg['pretrained'] = None

cfg = dict()

cfg['train_cfg'] = train_cfg
cfg['model_cfg'] = model_cfg

