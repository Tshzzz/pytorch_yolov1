#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:54:28 2018

@author: vl-tshzzz
"""

import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import numpy as np

root_dir = './'


sets = [('person','trainval_all')]


classes = ["person"]


def convert_bb(size,b):
    x = (b[0] + b[2])/2.0
    y = (b[1] + b[3])/2.0

    dw = 1./size[0]
    dh = 1./size[1]
    
    x = x*dw
    y = y*dh
    w = np.sqrt((b[2]-b[0])*dw)
    h = np.sqrt((b[3]-b[1])*dh)
    
    return(x,y,w,h)
    
    


def convert_xml(file_path,out_file):
    out_file = open(out_file,'w')
    tree=ET.parse(file_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        
        bb = convert_bb((w,h), b)
        out_file.write(str(cls_id) + "," + ",".join([str(a) for a in bb]) + '\n')
        
    out_file.close()



for data_ in sets:
    

    if not os.path.exists(root_dir+'VOC%s/Label/'%(data_[0])):
        os.makedirs(root_dir+'VOC%s/Label/'%(data_[0]))
            
    name_list = open(root_dir+'VOC%s/ImageSets/Main/%s.txt'%(data_[0],data_[1])).read().strip().split()
        
    print(len(name_list))
    name_list = tqdm(name_list)
    data_list = open('VOC%s_%s.txt'%(data_[0],data_[1]),'w')

    file_writer = ''
    for i,xml_name in enumerate(name_list):

        file_path = root_dir+'VOC%s/Annotations/%s.xml'%(data_[0],xml_name)
        label_file = root_dir + 'VOC%s/Label/%s.txt'%(data_[0],xml_name)
        img_file = root_dir + 'VOC%s/JPEGImages/%s.jpg'%(data_[0],xml_name)
        convert_xml(file_path,label_file)
            
            
        file_writer += img_file+' '+label_file+'\n'
        if i % 10000 == 0:    
            data_list.write(file_writer)       
            file_writer = ''
        
    
    data_list.close()
    
    
    
    
    
    
    