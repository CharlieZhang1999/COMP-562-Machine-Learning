#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 2020
@author: Charlie Zhang(djogem)
@assignment: COMP 562 final project
"""

import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset


from PIL import Image


def pil_loader(image_path):
    with open(image_path, 'rb') as f:
        return Image.open(f).convert('RGB')

#%%
#Custom Alexnet       
class CustomTrainDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: the dataset folder. The folder should be structured as:
                label1/image/500 .JPEG images
                label2/image/500 .JPEG images
                ...
                label200/image/500 .JPEG images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        

        #TODO: implement the needed initialization

        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        print(data_dir)
        for _root, _dirs, files in sorted(os.walk(data_dir)):
          for directory in _dirs:
            idx = self.class_to_idx[directory]
            directory_path = os.path.join(data_dir, directory)
            all_images_path = os.path.join(directory_path, 'images')
            print("one of 200 directory paths:", directory_path)
            #directory_path = os.path.join(data_dir, 'images')
            for curr_root, _subdirs, subfiles in sorted(os.walk(all_images_path)):
              #for cur_cur_root, _subsubdir, subsubfile in 
              for image_name in subfiles:
                image_path = os.path.join(curr_root, image_name)
                self.samples.append((image_path, idx))
          break


          

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Don't forget to apply the transform before returning."""
        path, target = self.samples[idx]
        sample = pil_loader(path)
        if self.transform is not None:  
            sample = self.transform(sample)

        return sample, target
        

class CustomValDataset(Dataset):
    """This is the Validation Dataset.

    Use this as a reference to implement you CustomTrainDataset.
    Remember because the training data and validation data are structured differently,
    you shouldn't directly use the code below. You should design it according to the
    training data folder's structure.
    """
    def __init__(self, data_dir, transform=None):
        self.root = os.path.normpath(data_dir)
        self.transform = transform

        classes_file = os.path.join(self.root, '../wnids.txt')

        self.classes = []
        with open(classes_file, 'r') as f:
            self.classes = f.readlines()
            for i in range(len(self.classes)):
                self.classes[i] = self.classes[i].strip()
        self.classes.sort()

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        images_y_file = os.path.join(self.root, 'val_annotations.txt')
        with open(images_y_file, 'r') as f:
            lines = f.readlines()

        self.class_dict = {}
        for line in lines:
            cols = line.split('\t')
            if len(cols) < 2:
                continue
            img_filename = cols[0].strip()
            img_class = cols[1].strip()
            self.class_dict[img_filename] = img_class

        self.samples = []
        images_dir = os.path.join(self.root, 'images')
        for _root, _dirs, files in sorted(os.walk(images_dir)):
            for image_name in files:
                image_path = os.path.join(_root, image_name)
                c = self.class_dict[image_name]
                idx = self.class_to_idx[c]
                self.samples.append((image_path, idx))


    def __getitem__(self, i):
        path, target = self.samples[i]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)
    

#%%
#Custom Alexnet
class CustomAlexnet(nn.Module):
    """Alexnet implementation for comp 590.
    
    Remember, you are required to implement alexnet using
    nn.Conv2d instead of nn.Linear.
    Failing to do that will lead to 0 points for this task.
    """
    def __init__(self, num_classes=200):
        super(CustomAlexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), 
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(256, 4096, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 200, kernel_size=1),
        )
        

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x
    

