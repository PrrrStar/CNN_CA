

import numpy as np

import torch
import torch.utils.data
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F

#from torchvision import datasets, models, transforms
import torch.optim as optim

#import matplotlib.pyplot as plt

import os

import torchvision.transforms as T

import torchvision

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

class loadImg(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
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
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

#        if self.transforms is not None:
 #           img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
class Model():
    def __init__ (self):
        

        cfg = Struct()
        cfg.batch_size =20
        cfg.train_epochs = 100
        cfg.num_worker = 0
        cfg.test_epochs = 1
        cfg.test_rate = 10
        
        cfg.optimizer = 'adam'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device,"에서 학습중 ...")
        
        self.cfg = cfg
        
        self.net = Net().to(self.device)
    
    def get_transform(self, train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
      
    def saveModel(self):
        print('Save model...')
        torch.save(self.ndet.state_dict(), 'model.pth')
        
    def loadModel(self):
        self.net.load_state_dict(torch.load('model.pth'))
        
    def get_model_instance_segmentation(num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))        
    
    def train(self):
        
  #      test_res, tmp_res, best_epoch =0, 0, 0
        
        self.net.train()
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
        train_set = loadImg('PennFudanPed', transform)
        test_set = loadImg('PennFudanPed', transform)
        
        # split the dataset in train and test set
        torch.manual_seed(1)
        indices = torch.randperm(len(train_set)).tolist()
        train_set = torch.utils.data.Subset(train_set, indices[:-50])
        test_set = torch.utils.data.Subset(test_set, indices[-50:])
        
        # define training and validation data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=2, shuffle=True, num_workers=self.cfg.num_worker,collate_fn=self.collate_fn)
        
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=self.cfg.num_worker,collate_fn=self.collate_fn)
            

        criterion = nn.CrossEntropyLoss().to(self.device)
    
        if self.cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr= 0.0001)
        elif self.cfg.optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.net.parameters(), lr =1.0, rho = 0.9, eps=1e-06, weight_decay=0)
        else :
            optimizer = optim.SGD(self.net.parameters(), lr = 0.0001, momentum=0.9)
            
        
        for epoch in range(self.cfg.train_epochs):
            
            train_loss, running_loss = 0, 0
            
            for i, data in enumerate(train_loader,0):
                
                
                inputs, labels = torch.tensor(data, dtype= torch.float64)
                
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
        
            train_loss = running_loss / len(train_loader)
            print('CE of the network on the traintset: %.6f' % (train_loss))
    def test(self):
        #set Test mode
        self.net.eval()
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        num_classes =2
        test_loss,running_loss =0, 0
        
        for epoch in range(self.cfg.test_epochs):
            for data in test_loader:
                inputs, labels = data
                inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
                
                outputs = self.net(inputs).to(self.device)
                
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
    #model.loadData()
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
