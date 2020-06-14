import torch
import numpy as np
import pandas as pd
import torch.optim as optim

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device,"에서 학습중 ...")


def csv2list(filePath):
    df = pd.read_csv(filePath)
    return df

file = csv2list("example.csv")

file['color'] = np.where(file['classes']==0,'blue',
    np.where(file['classes']==1,'red', 'yellow'))

x_col = torch.FloatTensor(file['x'].values)
y_col = torch.FloatTensor(file['y'].values)

#plt.scatter(x_col,y_col,c=file['color'])
#plt.show()

x_col = x_col.view(100,1)
y_col = y_col.view(100,1)

x = torch.cat([x_col,y_col], dim=1).to(device)
y = torch.FloatTensor(file['classes'].values).to(device)

x_train = torch.FloatTensor(x)
y_train = torch.FloatTensor(y)
y_train = y_train.view(100,1)

loss_values =[]
nb_epochs = 100000
def loss_func_graph(nb_epochs):
	epoch = str(nb_epochs)
	title = 'Epoch'+epoch+'_Loss.png'
	plt.plot(loss_values)
	plt.title('loss_value')
	plt.xlabel('Epochs')
	plt.ylabel('loss')
	plt.savefig(title)
	plt.show()

def GradientDescentAlgorithm():
    W = torch.rand((2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    def forward(X):
        return X*W
    
    def cross_entropy(hypothesis, y_train):
        return -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()
      
    def gradient(x_train,y_train):
        return 2 * torch.mean((W * x_train - y_train) * x_train)
    
    def sigmoid(x_train):
        return torch.sigmoid(x_train.matmul(W) + b)
     
    optimizer = optim.SGD([W, b], lr=1)

    for epoch in range(nb_epochs + 1):
        train_loss=0.0 
        hypothesis = sigmoid(x_train)
        cost = cross_entropy(hypothesis, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_loss += cost.item()
        loss_values.append(train_loss)
        
        if epoch % 10 == 0:
            pred = hypothesis >= torch.FloatTensor([0.5])
            correct = pred.float() == y_train
            accuracy = correct.sum().item() / len(correct) 
            print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}'.format(epoch, nb_epochs, cost.item(), accuracy *100))
GradientDescentAlgorithm()
loss_func_graph(nb_epochs)
