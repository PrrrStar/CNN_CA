import torch
import numpy as np

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms


num_worker = 0
batch_size = 20

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers = num_worker)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers = num_worker)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



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
        self.conv2 = nn.Conv2d(16, 32 ,kernel_size = 3, padding=1)       #input : 16, output : 32, kernel size = 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding=1)       #input : 32, output : 64, kernel size = 3
      
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500, 10)        
        self.dropout = nn.Dropout(0.25)         #dropout 25%

    def forward(self, x):
        x = F.relu(self.conv1(x))
#        print("conv1\t",x.shape)
        x = self.pool(x)
#        print("max1\t",x.shape)
        x = F.relu(self.conv2(x))
#        print("conv2\t",x.shape)
        x = self.pool(x)
#        print("max2\t",x.shape)
        x = F.relu(self.conv3(x))
#        print("conv3\t",x.shape)
        x = self.pool(x)
#        print("max3\t",x.shape)
        
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
#        print("drop1\t",x.shape)
        x = F.relu(self.fc1(x))                 #output size = 500
#        print("fc1\t",x.shape)
        x = self.dropout(x)
#        print("drop2\t",x.shape)        
        x = self.fc2(x)
#        print("fc2\t",x.shape)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,"에서 학습중 ...")
model = Net().to(device)


print(model)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()           
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


n_epochs = 50

model.train()
loss_values = []

for epoch in range(n_epochs):
    train_loss = 0.0
    
    #train model
    for i, data in enumerate(train_loader,0):
        
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()           #all param -> zero
        
        #forward pass : compute predicted output
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        loss_values.append(train_loss/len(train_loader))
        if i % 2000 == 1999:
            print('[%d, %5d]\tloss : %.3f'%
                  (epoch +1, i+1, train_loss/2000))

print("Finished!!")
print("loss values : ",loss_values)
plt.plot(loss_values)
plt.title('loss_values')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_value.png')
plt.show()

'''
모델 저장
'''
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)


'''
Sample Data로 Test
'''
dataiter = iter(test_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


'''
어떻게 신경망이 예측했는지 확인

출력은 10개 분류 각각에 대한 값으로 나타납니다. 
어떤 분류에 대해서 더 높은 값이 나타난다는 것은, 
신경망이 그 이미지가 해당 분류에 더 가깝다고 생각한다는 것입니다. 
따라서, 가장 높은 값을 갖는 인덱스(index)를 뽑아보겠습니다
'''
model = Net()
model.load_state_dict(torch.load(PATH))
outputs = model(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))



'''
전체 데이터 셋에 대해서 동작
'''
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


'''
어떤걸 잘 분류했는 지 정확도 테스트
'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
        




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
