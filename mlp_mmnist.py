import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)



num_worker = 0
batch_size = 20

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=False, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers = num_worker)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers = num_worker)

import matplotlib.pyplot as plt

dataiter = iter(train_loader)
#batch size 만큼 random 하게 데이터 뽑음

images, labels = dataiter.next()
images = images.numpy()
print(images.shape)


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden1 = 512
        hidden2 = 512
        self.fc1 = nn.Linear(28*28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28) #모든 애들의 차원을 28by28로 바꿔줌
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Net()
model = model.cuda()

print(model)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


n_epochs = 50

model.train()

for epoch in range(n_epochs):
    train_loss = 0.0
    
    #train model
    for data, target in train_loader:
        optimizer.zero_grad()           #all param -> zero
        
        data = data.to(device)
        target = target.to(device)
        #forward pass : compute predicted output
        output = model(data)
        output = output.to(device)
        loss = criterion(output, target)
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()*data.size(0) #compute loss for all data in batch
    #trian batch_sampler &
    #train loader  = num of all data  /  batch size
    
    #train sampler = num of all data 
    #trian batch_sampler = 
    train_loss = train_loss/len(train_loader.sampler) #train load's sampler
    
    print("Epoch: {}\tTraining Loss: {:.6f}".format(epoch+1, train_loss))
        

class_correct = list(0. for i in range(10))
class_total_number = list(0. for i in range(10))

model.eval()

for data, target in test_loader:
    output = model(data)
    val, pred = torch.max(output,1)
    correct= pred.eq(target)
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label]+= correct[i].item()
        class_total_number[lable] +=1
        
for i in range(10):
    if class_total_number[i] > 0:
        print(i, " : ",100*class_correct[i]/class_total_number[i], '(',np.sum(class_correct[i]),'/',np.sum(class_total_number[i]),')')
    