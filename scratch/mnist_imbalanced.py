import torch
import torch.nn as nn
import numpy as np
import math
import random
from torch import optim
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


#Data loading
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)


#Create a two-class-dataset of with just 1 and 7
class_skew = 20
classes = [5,6]
reduction_factor = 10
batch_size = 16

#Indices for the data with labels either 1 or 7
idx_class_one = train_data.targets== classes[0]
idx_class_two = train_data.targets== classes[1]


#Sub_subsample to get reduce the second class examples by a factor of class_skew*reduction_factor
indices_for_class_two = [i for i, x in enumerate(idx_class_two) if x]
subsampled_indices = (random.sample(indices_for_class_two, int(len(indices_for_class_two)/(class_skew*reduction_factor))))
for i in range(len(idx_class_two)):
    if i not in subsampled_indices:
        idx_class_two[i] = False
#Sub_subsample to get reduce the first class examples by a factor of reduction_factor        
indices_for_class_one = [i for i, x in enumerate(idx_class_one) if x]
subsampled_indices = (random.sample(indices_for_class_one, int(len(indices_for_class_one)/(reduction_factor))))
for i in range(len(idx_class_one)):
    if i not in subsampled_indices:
        idx_class_one[i] = False

print(sum(idx_class_one))
print(sum(idx_class_two))
idx = idx_class_one+idx_class_two
#Training data
train_data.targets = train_data.targets[idx]
#Converting classes to 0 and 1
idx_class_one = train_data.targets== classes[0]
idx_class_two = train_data.targets== classes[1]
train_data.targets[idx_class_one] = (train_data.targets[idx_class_one]/classes[0]-1).type(torch.LongTensor)
train_data.targets[idx_class_two] = (train_data.targets[idx_class_two]/classes[1]).type(torch.LongTensor)
train_data.targets = train_data.targets
train_data.data = train_data.data[idx]




#Test data with just 1 and 
#Indices for the data with labels either 1 or 7
idx_class_one = test_data.targets== classes[0]
idx_class_two = test_data.targets== classes[1]
idx = idx_class_one+idx_class_two

#Test data
test_data.targets = test_data.targets[idx]
idx_class_one = test_data.targets== classes[0]
idx_class_two = test_data.targets== classes[1]
test_data.targets[idx_class_one] = (test_data.targets[idx_class_one]/classes[0]-1).type(torch.LongTensor)
test_data.targets[idx_class_two] = (test_data.targets[idx_class_two]/classes[1]).type(torch.LongTensor)
test_data.targets = test_data.targets
test_data.data = test_data.data[idx]



loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
}
loaders

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 2 classes
        self.out = nn.Linear(32 * 7 * 7, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

cnn = CNN()

class PolyTailLoss(torch.nn.Module):
    def __init__(self, alpha):
        self.alpha = alpha
        super(PolyTailLoss, self).__init__()

    def forward(self, logits, target):
        margin_vals = ((logits[:,1]-logits[:,0])*(2*target-1)).view(-1)
        return self.margin_fn(margin_vals)

    def margin_fn(self, margin_vals):
        indicator = margin_vals <= 1
        scores = torch.zeros_like(margin_vals)
        inv_part = torch.pow(margin_vals, -1*self.alpha)
        logit_inner = -1*margin_vals
        logit_part = (torch.log(torch.exp(logit_inner)+1))/math.log(1+math.exp(-1))
        scores[indicator] = logit_part[indicator]
        scores[~indicator] = inv_part[~indicator]
        return scores

#Training parameters
#Default parameters for full gradients, epochs = 1000, lr = 0.01
#Default parameters for batch_size = 32, epochs = 1000, lr = 0.01
#Default parameters for batch_size = 16, epochs = 1000, lr = 0.01
num_epochs = 1000
#loss_func = nn.CrossEntropyLoss(reduce=None, reduction='none')
loss_func = PolyTailLoss(alpha=2)
optimizer = optim.SGD(cnn.parameters(), lr = 0.01)

def train(num_epochs, cnn, loaders):
    
    cnn.train()
    # Train the model
    total_step = len(loaders['train'])
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            idx_class_1 = b_y == 1
            weights = torch.ones_like(b_y, requires_grad=False, dtype=torch.float64)
            weights[idx_class_1] = 20*torch.ones_like(weights[idx_class_1])
            weights = weights/5
            output = cnn(b_x)[0]
            #loss = loss_func(output, b_y).mean()
            loss = torch.mean(loss_func(output, b_y)*weights)
            # clear gradients for this training step   
            optimizer.zero_grad()           
            loss.backward()               
            optimizer.step() 
            
        if (epoch+1) % 5 == 0:
            cnn.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                correct_te = 0
                total_te = 0
                frac_pos = 0
                correct_cl_0 = 0
                total_cl_0 = 0
                correct_cl_1 = 0
                total_cl_1 = 0
                for images, labels in loaders['train']:
                    train_output, last_layer = cnn(images)
                    pred_y = torch.max(train_output, 1)[1].data.squeeze()
                    correct += (pred_y == labels).sum().item() 
                    total += float(labels.size(0))
                accuracy = correct/total
                for images, labels in loaders['test']:
                    test_output, last_layer = cnn(images)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    #Total test accuracy
                    correct_te += (pred_y == labels).sum().item()
                    total_te += float(labels.size(0))
                    #fraction of labels predicted to be positive
                    frac_pos += torch.sum(pred_y).item()
                    #class_0_accuracy
                    idx_cl_0 = labels == 0
                    correct_cl_0 += (pred_y[idx_cl_0] == labels[idx_cl_0]).sum().item() 
                    total_cl_0 += float(labels[idx_cl_0].size(0))
                    #class_1_accuracy
                    idx_cl_1 = labels == 1
                    correct_cl_1 += (pred_y[idx_cl_1] == labels[idx_cl_1]).sum().item() 
                    total_cl_1 += float(labels[idx_cl_1].size(0))
            
                test_accuracy = correct_te/total_te
                fraction_pos = frac_pos/total_te
                accuracy_cl_0 = correct_cl_0/total_cl_0
                accuracy_cl_1 = correct_cl_1/total_cl_1
                difference = abs(accuracy_cl_0-accuracy_cl_1)
                
            print ('Epoch [{}/{}], Loss: {:.4f}, Tr Acc: {:,.4f}, Tt Acc : {:,.4f}, Frac Pos : {:,.4f}, Test C0 : {:,.4f}, Test C1 : {:,.4f}, Difference : : {:,.4f}' 
                   .format(epoch + 1, num_epochs,  loss.item(),accuracy, test_accuracy, fraction_pos, accuracy_cl_0,accuracy_cl_1,difference))
            
        
train(num_epochs, cnn, loaders)