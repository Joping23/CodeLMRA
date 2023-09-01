#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:58:14 2023

@author: idriouich
"""

# -*- coding: utf-8 -*-
import torch 
import torchvision
from torchvision import transforms
from PIL import Image
import glob
import os
import json
num_class = []
images = []

for worker_id in range(0,10) : 
    num_class = []
    images = []
    image_list = []
    for filename in glob.glob('ORL/'+str(worker_id)+ '/*.jpg'): #assuming gif
        num_class.append(int(filename.split('_')[1].split('.')[0])-1)
        im=Image.open(filename)
        image_list.append(im)
        
    
    data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            
            transforms.Normalize(mean=[0.485,],
                                 std=[0.229,])
        ])
    
    
    for i in image_list :
        images.append(data_transform(i).tolist())
     
    train_dir = "train"
    test_dir = "test"
    train_file = os.path.join(train_dir, str(worker_id) +".json") 
    test_file = os.path.join(test_dir, str(worker_id) +".json") 
    json_data_train = {"x": images, "y": num_class, "num_classes": 41}
    json_data_test = {"x": images, "y": num_class, "num_classes": 41}
    
    with open(train_file, 'w') as outfile:
        json.dump(json_data_train, outfile)
    with open(test_file, 'w') as outfile:
        json.dump(json_data_test, outfile)
    

   
        
        
train_dir = "train"
test_dir = "test"
train_file = os.path.join(train_dir, "train"+".json")
test_file = os.path.join(test_dir, "test"+".json") 
json_data_train = {"x": images, "y": num_class, "num_classes": 41}
json_data_test = {"x": images, "y": num_class, "num_classes": 41}

with open(train_file, 'w') as outfile:
    json.dump(json_data_train, outfile)
with open(test_file, 'w') as outfile:
    json.dump(json_data_test, outfile)
    
    
    

   
        
        

train_file = os.path.join( "all_data"+".json")
json_data_train = {"x": images, "y": num_class, "num_classes": 41}
json_data_test = {"x": images, "y": num_class, "num_classes": 41}

with open(train_file, 'w') as outfile:
    json.dump(json_data_train, outfile)
with open(train_file, 'w') as outfile:
    json.dump(json_data_test, outfile)