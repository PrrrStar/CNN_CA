import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms


num_worker = 2
batch_size = 20

transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers = num_worker)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers = num_worker)

import matplotlib.pyplot as plt

dataiter = iter(train_loader)
#batch size 만큼 random 하게 데이터 뽑음

images, labels = dataiter.next()
images = images.numpy()
print("image_size",images.shape)


import torch.nn as nn
import torch.nn.functional as F


def cnn_output_tensor_size(input_size, kernel_size, stride, padding):
    output_size = (input_size - kernel_size + 2*padding + stride) / stride
    return output_size

def maxPool_output_tensor_size(input_size, stride, pooling_size):
    output_size = (input_size - pooling_size + stride) / stride
    return output_size

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding=1)        #input : 3, output : 16, kernel size = 3
        self.conv2 = nn.Conv2d(16, 32 ,kernel_size = 3, padding=2)       #input : 16, output : 32, kernel size = 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding=1)       #input : 32, output : 64, kernel size = 3
      
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 1)
        
        self.fc1 = nn.Linear(32*32*5, 500)
        self.fc2 = nn.Linear(500, 10)        
        self.dropout = nn.Dropout(0.25)         #dropout 25%

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print("conv1\t",x.shape)
        x = self.pool(x)
        print("max1\t",x.shape)
        x = F.relu(self.conv2(x))
        print("conv2\t",x.shape)
        x = self.pool(x)
        print("max2\t",x.shape)
        x = F.relu(self.conv3(x))
        print("conv3\t",x.shape)
        x = self.pool(x)
        print("max3\t",x.shape)
        
        x = x.view(-1, 5120)
        x = self.dropout(x)
        print("drop1\t",x.shape)
        x = F.relu(self.fc1(x))                 #output size = 500
        print("fc1\t",x.shape)
        x = self.dropout(x)
        print("drop2\t",x.shape)        
        x = self.fc2(x)
        print("fc2\t",x.shape)
        return x

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)
    model = Net().to(device)
    
else:
    device = torch.device("cpu")
    print(device)
    model = Net()


print(model)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()           
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


n_epochs = 50

model.train()

for epoch in range(n_epochs):
    train_loss = 0.0
    
    #train model
    for i, data in enumerate(train_loader,0):
        
        inputs, labels = data
        
        optimizer.zero_grad()           #all param -> zero
        
        #forward pass : compute predicted output
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d]\tloss : #.3f'%
                  (epoch +1, i+1, train_loss/2000))

print("Finished!!")
        
        
'''
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
        class_total_number[label] +=1
        
for i in range(10):
    if class_total_number[i] > 0:
        print(i, " : ",100*class_correct[i]/class_total_number[i], '(',np.sum(class_correct[i]),'/',np.sum(class_total_number[i]),')')
'''  