import torch

from models import *

import cv2
image = cv2.imread('tools/val_3.jpg')

image = image.transpose((2, 0, 1)) / 255.
image = torch.from_numpy(image).float()
image = image.unsqueeze(dim=0)


cfg = dict()
cfg['class_num'] = 1
cfg['backbone'] = 34
checkpoint = torch.load('./experiment/Cervical/Basenet/best_model.pth')['model']
data_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

model = base_detector(cfg)
model.load_state_dict(data_dict)

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(100) ]
output_names = [ "output1" ]

torch.onnx.export(model, image, "res34.onnx",
                  verbose=True, input_names=input_names,
                  output_names=output_names)

