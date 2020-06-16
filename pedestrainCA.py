# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:31:18 2020

@author: rlawl
"""

import torch
#import torchvision

import torch.nn as nn
import torch.nn.functional as F

#from torchvision import datasets, models, transforms
import torch.optim as optim

import numpy as np
#import matplotlib.pyplot as plt

import os

import transforms as T

from PIL import Image

from torch.autograd import Variable
class Struct(object): pass

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 5, stride = (2,2))
        self.conv2 = nn.Conv2d(24, 36, 5, stride = (2,2))
        self.conv3 = nn.Conv2d(36, 48, 5, stride = (2,2))
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.drop = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(64*3*13, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10,1)
        
        
        
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = self.drop(x)
        print(x.size())
        
        x = x.view(-1, 64*3*13)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
class Model():
    def __init__ (self):
        

        cfg = Struct()
        cfg.batch_size =100
        cfg.train_epochs = 100
        cfg.num_worker = 4
        cfg.test_epochs = 1
        cfg.test_rate = 10
        
        cfg.optimizer = 'adam'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device,"에서 학습중 ...")
        
        self.cfg = cfg
        
        self.net = Net().to(self.device)
    
    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)


    def loadImg(object):
        def __init__(self, root, transforms):
            self.root = root
            self.transforms = transforms
            
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
            
        def __getitem__(self, idx):
            img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
            
            #컬러 인스턴스로 객체 구별"
            img = Image.open(img_path).convert("RGB")
            
            mask = Image.open(mask_path)
            mask = np.array(mask)
            
            obj_ids = np.unique(mask)
            
            #[0] = 배경.
            obj_ids = obj_ids[1:]
            
            masks = mask == obj_ids[:, None, None]
            
            num_objs = len(obj_ids)
            boxes = []
            
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            
            image_id = torch.tensor([idx])
            
            #영역 넓이
            area = (boxes[:,3] - boxes[:,1])*(boxes[:,2]-boxes[:,0])
            
            iscrowd = torch.zeros((num_objs,),dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target
        
        def __len__(self):
            return len(self.imgs)
        
    def loadData(self):
        train_set = self.loadImg("PennFudanPed",self.get_transform(train=True))
        test_set = self.loadImg("PennFudanPed",self.get_transform(train=False))
        
        indices = torch.randperm(len(train_set)).tolist()
        
        train_set = torch.utils.data.Subset(train_set, indices=[:50])
        test_set = torch.utils.data.Subset(test_set, indices=[50:])
        

        self.train_loader = torch.utils.data.DataLoader(
                train_set, batch_size= self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        self.test_loader = torch.utils.data.DataLoader(
                test_set, batch_size= self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        
    def saveModel(self):
        print('Save model...')
        torch.save(self.ndet.state_dict(), 'model.pth')
        
    def loadModel(self):
        self.net.load_state_dict(torch.load('model.pth'))
        
    def get_model_instance_segmentation(num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn()
        
    def train(self):
        
        train_set = self.loadData("PennFudanPed",self.get_transform(train=True))
        train_loader = torch.utils.data.DataLoader(
                train_set, batch_size= self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        
  #      test_res, tmp_res, best_epoch =0, 0, 0
        
        self.net.train()
        

        criterion = nn.CrossEntropyLoss().to(self.device)
    
        if self.cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr= 0.0001)
        elif self.cfg.optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.net.parameters(), lr =1.0, rho = 0.9, eps=1e-06, weight_decay=0)
        else :
            optimizer = optim.SGD(self.net.parameters(), lr = 0.0001, momentum=0.9)
            
        
        for epoch in range(self.cfg.train_epochs):
            train_loss, running_loss = 0.0
            
            for i, data in enumerate(self.train_loader,0):
                inputs, labels = data
                
                inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
                    
                optimizer.zero_grad()
                
                outputs = self.net(inputs).to(self.device)
                
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i %10 == 0:
                    print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / (i+1)))
        
            train_loss = running_loss / len(self.train_loader)
            print('CE of the network on the traintset: %.6f' % (train_loss))
    def test(self):
        #set Test mode
        self.net.eval()
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        num_classes =2
        test_loss,running_loss =0, 0
        
        for epoch in range(self.cfg.test_epochs):
            for data in self.test_loader:
                inputs, labels = data
                inputs, labes = Variable(inputs).to(self.device), Variable(labels).to(self.device)
                
                outputs - self.net(inputs).to(self.device)
                
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                del loss
                
        if (self.cfg.test_epoch>0):
             test_loss = running_loss/(len(self.test_loader)*self.cfg.test_epochs)
        print('CE of the network on the test_set: %.6f'%(test_loss))
        
        self.net.train()
        
        return test_loss
    
if __name__ =='__main__':
    model = Model()
    model.loadData()
    model.train()
    
        
         
        
        
        
        
      #  model = get_model_instance_segmentation(num_classes).to(self.device)
        
        
                
     
'''
             
# 학습 시
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# 추론 시
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
'''
