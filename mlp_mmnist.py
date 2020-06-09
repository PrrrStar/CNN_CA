import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

num_worker = 0
batch_size = 16

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers = num_workers)

import matplotlib.pyplot as plt

dataiter = iter(train_loader)
#batch size 만큼 random 하게 데이터 뽑음

images, labels = dataiter.next()
images = images.numpy()
print(images.shape())


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden1 = 512
        hidden2 = 512
        self.fc1 = nn.Linear(28*28, hidden1)
        self.fc2 = nn.Linear(hidden, hidden2)
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
print(model)





