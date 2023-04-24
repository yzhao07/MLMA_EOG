import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
import cv2
import numpy as np

class Dataset():
    def __init__(self, stroke,img_path, transform=None):
        self.stroke = stroke
        self.img_path = img_path
        self.transform = transform


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        data = self.stroke[index]
        img = cv2.imread(self.img_path[index])
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.img_path[index].split("_")[-1].split(".")[0])-1

        return data,img,label
