import torch.nn as nn
import torch.nn.functional as F
import torch

class Model_name(nn.Module):
    # __init__ allows you to write the basic component of the model
    def __init__(self):
        super(Model_name,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,3,stride = 1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(4),
            nn.BatchNorm1d(16,momentum=0.99), 
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,16,3,stride = 1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(4),
            nn.BatchNorm1d(16,momentum=0.99), 
            nn.Flatten()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(1088, 512)
            nn.ReLU()
            nn.Dropout(p=0.5)
            nn.Linear(512, 150)
            #('output',nn.Softmax(dim=1))     
        )
        self.flatten = nn.Flatten()

    def forward(self, data,img):
        data = self.flatten(data)
        img = self.layer1(img)
        img = self.layer2(img)
        x = torch.cat((img, data), dim=1)
        y = self.classifier2(x)

        return y
    
class Model_name_img(nn.Module):
    # __init__ allows you to write the basic component of the model
    def __init__(self):
        super(Model_name,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,3,stride = 1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(4),
            nn.BatchNorm1d(16,momentum=0.99), 
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,16,3,stride = 1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(4),
            nn.BatchNorm1d(16,momentum=0.99), 
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(576, 512)
            nn.ReLU()
            nn.Dropout(p=0.5)
            nn.Linear(512, 150)
  
        )
        
    def forward(self, data,img):
        img = self.layer1(img)
        img = self.layer2(img)
        y = self.classifier(img)
        return y