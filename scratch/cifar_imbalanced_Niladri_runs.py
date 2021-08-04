import torch
import numpy as np
import math
import random

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

####Load the data####
transform = transforms.Compose(
    [ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)


train_data.data = torch.FloatTensor(train_data.data)
train_data.targets = torch.LongTensor(train_data.targets)
test_data.data = torch.FloatTensor(test_data.data)
test_data.targets = torch.LongTensor(test_data.targets)


# Create a two-class-dataset of with just classes specified by the variable classes
class_skew = 5
classes = [3, 5]
reduction_factor = 5

# Indices for the data with labels either of the classes
idx_class_one = torch.BoolTensor(
    [x == classes[0] for i, x in enumerate(train_data.targets)]
)
idx_class_two = torch.BoolTensor(
    [x == classes[1] for i, x in enumerate(train_data.targets)]
)


# Sub_subsample to get reduce the second class examples by a factor of class_skew*reduction_factor
indices_for_class_two = [i for i, x in enumerate(idx_class_two) if x]
subsampled_indices = random.sample(
    indices_for_class_two,
    int(len(indices_for_class_two) / (class_skew * reduction_factor)),
)
for i in range(len(idx_class_two)):
    if i not in subsampled_indices:
        idx_class_two[i] = False
# Sub_subsample to get reduce the first class examples by a factor of reduction_factor
indices_for_class_one = [i for i, x in enumerate(idx_class_one) if x]
subsampled_indices = random.sample(
    indices_for_class_one, int(len(indices_for_class_one) / (reduction_factor))
)
for i in range(len(idx_class_one)):
    if i not in subsampled_indices:
        idx_class_one[i] = False

print("Number of samples in class one", sum(idx_class_one).item())
print("Number of samples in class two", sum(idx_class_two).item())
idx = torch.logical_or(idx_class_one, idx_class_two)

# Training data
train_data.targets = train_data.targets[idx]
# print('Length of train_data', train_data.targets.size())
# Converting classes to 0 and 1
idx_class_one = torch.BoolTensor(
    [x == classes[0] for i, x in enumerate(train_data.targets)]
)
idx_class_two = torch.BoolTensor(
    [x == classes[1] for i, x in enumerate(train_data.targets)]
)
train_data.targets[idx_class_one] = (
    train_data.targets[idx_class_one] / classes[0] - 1
).type(torch.LongTensor)
train_data.targets[idx_class_two] = (
    train_data.targets[idx_class_two] / classes[1]
).type(torch.LongTensor)
train_data.targets = train_data.targets.tolist()
train_data.targets = [int(item) for item in train_data.targets]
train_data.data = train_data.data[idx].numpy().astype(np.uint8)
print("Total number of training samples", len(train_data.targets))
# print(train_data.data[0].dtype)

# batch_size = 16

# Test data
# Indices for the data with labels either of the classes
idx_class_one = torch.BoolTensor(
    [x == classes[0] for i, x in enumerate(test_data.targets)]
)
idx_class_two = torch.BoolTensor(
    [x == classes[1] for i, x in enumerate(test_data.targets)]
)
idx = torch.logical_or(idx_class_one, idx_class_two)

# Test data
test_data.targets = test_data.targets[idx]
idx_class_one = torch.BoolTensor(
    [x == classes[0] for i, x in enumerate(test_data.targets)]
)
idx_class_two = torch.BoolTensor(
    [x == classes[1] for i, x in enumerate(test_data.targets)]
)
test_data.targets[idx_class_one] = (
    test_data.targets[idx_class_one] / classes[0] - 1
).type(torch.LongTensor)
test_data.targets[idx_class_two] = (test_data.targets[idx_class_two] / classes[1]).type(
    torch.LongTensor
)
test_data.targets = test_data.targets.tolist()
test_data.targets = [int(item) for item in test_data.targets]
test_data.data = test_data.data[idx].numpy().astype(np.uint8)


batch_size = len(train_data.targets)
# trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
# shuffle=True, num_workers=1)
loaders = {
    "train": torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=1
    ),
    "test": torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=1
    ),
}
loaders


####Definition of the architechture####
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnn = Net()

####Defining the loss function####
class PolyTailLoss(torch.nn.Module):
    def __init__(self, alpha):
        self.alpha = alpha
        super(PolyTailLoss, self).__init__()

    def forward(self, logits, target):
        margin_vals = ((logits[:, 1] - logits[:, 0]) * (2 * target - 1)).view(-1)
        return self.margin_fn(margin_vals)

    def margin_fn(self, margin_vals):
        indicator = margin_vals <= 1
        scores = torch.zeros_like(margin_vals)
        inv_part = torch.pow(margin_vals, -1 * self.alpha)
        logit_inner = -1 * margin_vals
        logit_part = (torch.log(torch.exp(logit_inner) + 1)) / math.log(
            1 + math.exp(-1)
        )
        scores[indicator] = logit_part[indicator]
        scores[~indicator] = inv_part[~indicator]
        return scores


####Training loop####

# Training parameters
num_epochs = 5000
iw_factor = 5
# loss_func = nn.CrossEntropyLoss(reduce=None, reduction='none')
loss_func = PolyTailLoss(alpha=2)
optimizer = optim.SGD(cnn.parameters(), lr=0.05)


def train(num_epochs, cnn, loaders):

    cnn.train()
    # Train the model
    total_step = len(loaders["train"])
    for epoch in range(num_epochs):
        for i, data in enumerate(loaders["train"], 0):
            # gives batch data, normalize x when iterate train_loader
            images, labels = data
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y
            idx_class_1 = b_y == 1
            weights = torch.ones_like(b_y, requires_grad=False, dtype=torch.float64)
            weights[idx_class_1] = iw_factor * torch.ones_like(weights[idx_class_1])
            weights = weights / iw_factor
            output = cnn(b_x)
            # loss = loss_func(output, b_y).mean()
            # print(output.size())
            # print(cnn(b_x).size())
            # print(loss_func(output, b_y).size())
            loss = torch.mean(loss_func(output, b_y) * weights)
            # clear gradients for this training step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
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
                for i, data in enumerate(loaders["train"], 0):
                    images, labels = data
                    train_output = cnn(images)
                    pred_y = torch.max(train_output, 1)[1].data.squeeze()
                    correct += (pred_y == labels).sum().item()
                    total += float(labels.size(0))
                accuracy = correct / total
                for i, data in enumerate(loaders["test"], 0):
                    images, labels = data
                    test_output = cnn(images)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    # Total test accuracy
                    correct_te += (pred_y == labels).sum().item()
                    total_te += float(labels.size(0))
                    # fraction of labels predicted to be positive
                    frac_pos += torch.sum(pred_y).item()
                    # class_0_accuracy
                    idx_cl_0 = labels == 0
                    correct_cl_0 += (pred_y[idx_cl_0] == labels[idx_cl_0]).sum().item()
                    total_cl_0 += float(labels[idx_cl_0].size(0))
                    # class_1_accuracy
                    idx_cl_1 = labels == 1
                    correct_cl_1 += (pred_y[idx_cl_1] == labels[idx_cl_1]).sum().item()
                    total_cl_1 += float(labels[idx_cl_1].size(0))

                test_accuracy = correct_te / total_te
                fraction_pos = frac_pos / total_te
                accuracy_cl_0 = correct_cl_0 / total_cl_0
                accuracy_cl_1 = correct_cl_1 / total_cl_1
                difference = abs(accuracy_cl_0 - accuracy_cl_1)

            print(
                "Epoch [{}/{}], Loss: {:.4f}, Tr Acc: {:,.4f}, Tt Acc : {:,.4f}, Frac Pos : {:,.4f}, Test C0 : {:,.4f}, Test C1 : {:,.4f}, Difference : : {:,.4f}".format(
                    epoch + 1,
                    num_epochs,
                    iw_factor * loss.item(),
                    accuracy,
                    test_accuracy,
                    fraction_pos,
                    accuracy_cl_0,
                    accuracy_cl_1,
                    difference,
                )
            )


train(num_epochs, cnn, loaders)
