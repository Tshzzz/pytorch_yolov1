#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:49:39 2018

@author: tshzzz
"""
from PIL import Image,ImageEnhance
import random
import numpy as np

def random_contract(Coeffi,img):
    return ImageEnhance.Contrast(img).enhance(Coeffi)
        
def random_Brightness(Coeffi,img):
    return ImageEnhance.Brightness(img).enhance(Coeffi)
    
def random_Color(Coeffi,img):
    return ImageEnhance.Color(img).enhance(Coeffi) 

def random_Sharpness(Coeffi,img):
    return ImageEnhance.Sharpness(img).enhance(Coeffi) 

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh,dh)

        
    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot


    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2

    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy


    sized = cropped.resize(shape,Image.NEAREST)
    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(bs, flip, dx, dy, sx, sy):

    new_bs = []
    #print(bs)
    for i in range(bs.shape[0]):
        x1 = bs[i][0]
        y1 = bs[i][1]
        x2 = bs[i][0] + bs[i][2]
        y2 = bs[i][1] + bs[i][3]
        
        x1 = min(0.99, max(0, x1 * sx - dx)) 
        y1 = min(0.99, max(0, y1 * sy - dy)) 
            
        x2 = max(0, min(0.999, x2 * sx - dx))
        y2 = max(0, min(0.999, y2 * sy - dy))
            
        
        bs[i][0] = x1
        bs[i][1] = y1 
        bs[i][2] = x2 - x1
        bs[i][3] = y2 - y1
        bs[i][4] = bs[i][4]
        if flip:
            bs[i][0] =  1 - bs[i][0] - bs[i][2]
                
        if bs[i][2] > 0 and bs[i][3] > 0:
            new_bs.append([bs[i]])
   
    new_bs = np.array(new_bs)
    new_bs = np.reshape(new_bs, (-1, 5))

    return new_bs


def norm_bb(b,size):
    x = b[:, 0:1]
    y = b[:, 1:2]

    dw = 1. / size[0]
    dh = 1. / size[1]

    x = (x * dw).clip(0.01, 0.99)
    y = (y * dh).clip(0.01, 0.99)
    w = ((b[:, 2:3] - b[:, 0:1]) * dw).clip(0.01, 0.99)
    h = ((b[:, 3:4] - b[:, 1:2]) * dh).clip(0.01, 0.99)

    return np.concatenate((x, y, w, h, b[:, 4:5]), axis=1)

def load_data_detection(imgpath,labpath ,shape, aug = True, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):

    img = Image.open(imgpath).convert('RGB')

    bs = np.loadtxt(labpath,delimiter=',') 
    bs = np.reshape(bs, (-1, 5))
    
    bs = norm_bb(bs,(img.width,img.height))
    
    if aug:
        img, flip, dx, dy, sx, sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    else:
        flip, dx, dy, sx, sy = False, 0, 0, 1, 1
        
    #print(dx, dy, sx, sy)
    label = fill_truth_detection(bs, flip, dx, dy, 1./sx, 1./sy)
    
    return img,label




