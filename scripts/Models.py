import torch
import torch.nn as nn

import os

import torchvision

class CNN(nn.Module):

    def __init__(self,Data):
        super(CNN,self).__init__()
        
        self.dataset = Data

        self.layer1 = nn.Sequential(
            nn.Conv2d(Data.num_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, Data.num_classes))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def save(self,algo_name,fold,epoch):
        
        path = "../models/"+self.dataset.dataset_name+"/CNN/"+algo_name+"/"+fold
        
        os.makedirs(path,exist_ok=True)
            
        torch.save(self.state_dict(), path+"/"+epoch+".pth")
        
    def clean(self,algo_name,fold):
        path = "../models/"+self.dataset.dataset_name+"/CNN/"+algo_name+"/"+fold
        
        if(os.path.exists(path)):
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))

class ResNet18(nn.Module):
    def __init__(self, Data):
        super(ResNet18,self).__init__()
        
        self.dataset = Data
        
        self.resnet = torchvision.models.resnet18()
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, Data.num_classes)

    def forward(self, x):
        return self.resnet(x) 
    
    def save(self,algo_name,fold,epoch):
        
        path = "../models/"+self.dataset.dataset_name+"/ResNet18/"+algo_name+"/"+fold
        
        os.makedirs(path,exist_ok=True)
            
        torch.save(self.state_dict(), path+"/"+epoch+".pth")
        
    def clean(self,algo_name,fold):
        path = "../models/"+self.dataset.dataset_name+"/ResNet18/"+algo_name+"/"+fold
        
        if(os.path.exists(path)):
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
    
    
class ResNet18_pretrainied(nn.Module):
    def __init__(self, Data):
        super(ResNet18_pretrainied,self).__init__()
        
        self.dataset = Data
        
        self.resnet = torchvision.models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, Data.num_classes)
        
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)
    
    def save(self,algo_name,fold,epoch):
        
        path = "../models/"+self.dataset.dataset_name+"/ResNet18_finetuning/"+algo_name+"/"+fold
        
        os.makedirs(path,exist_ok=True)
            
        torch.save(self.state_dict(), path+"/"+epoch+".pth")
        
    def clean(self,algo_name,fold):
        path = "../models/"+self.dataset.dataset_name+"/ResNet18_finetuning/"+algo_name+"/"+fold
                
        if(os.path.exists(path)):
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
