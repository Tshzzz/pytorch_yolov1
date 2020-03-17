
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

layer_configs = [
    # Unit1 (2)
    (32, 3, True),
    (64, 3, True),
    # Unit2 (3)
    (128, 3, False),
    (64, 1, False),
    (128, 3, True),
    # Unit3 (3)
    (256, 3, False),
    (128, 1, False),
    (256, 3, True),
    # Unit4 (5)
    (512, 3, False),
    (256, 1, False),
    (512, 3, False),
    (256, 1, False),
    (512, 3, True),
    # Unit5 (5)
    (1024, 3, False),
    (512, 1, False),
    (1024, 3, False),
    (512, 1, False),
    (1024, 3, False),
]

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()

    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b

    conv_weight = torch.from_numpy(buf[start:start + num_w])
    conv_model.weight.data.copy_(conv_weight.view_as(conv_model.weight))
    start = start + num_w

    return start
class conv_block(nn.Module):

    def __init__(self, inplane, outplane, kernel_size, pool, stride=1):
        super(conv_block, self).__init__()

        pad = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(inplane, outplane, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.act = nn.LeakyReLU(0.1)
        self.pool = pool  # MaxPool2d(2,stride = 2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        if self.pool:
            out = F.max_pool2d(out, kernel_size=2, stride=2)

        return out


class darknet_19(nn.Module):

    def __init__(self, cls_num=1000):
        super(darknet_19, self).__init__()
        self.class_num = cls_num
        self.feature = self.make_layers(3, layer_configs)

    def make_layers(self, inplane, cfg):
        layers = []

        for outplane, kernel_size, pool in cfg:
            layers.append(conv_block(inplane, outplane, kernel_size, pool))
            inplane = outplane

        return nn.Sequential(*layers)

    def load_weight(self, weight_file):
        print("Load pretrained models !")

        fp = open(weight_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        header = torch.from_numpy(header)
        buf = np.fromfile(fp, dtype=np.float32)

        start = 0
        for idx, m in enumerate(self.feature.modules()):
            if isinstance(m, nn.Conv2d):
                conv = m
            if isinstance(m, nn.BatchNorm2d):
                bn = m
                start = load_conv_bn(buf, start, conv, bn)

        assert start == buf.shape[0]

    def forward(self, x):

        output = self.feature(x)

        return output
