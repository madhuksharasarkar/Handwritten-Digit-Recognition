from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision


def load_data(train_batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((32,32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((32,32)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
    batch_size=test_batch_size, shuffle=True)

    return (train_loader, test_loader)

###############################################################################
############################ LeNet-5 ##########################################
###############################################################################

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


train_loss_lenet = []
test_loss_lenet = []
test_accuracy_lenet = []

def train(model, optimizer, epoch, train_loader, log_interval):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train set: Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item()))
            train_loss_lenet.append(loss.item())

def test(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += loss_fn(output, target).item()
        pred = np.argmax(output.data, axis=1)
        correct = correct + np.equal(pred, target.data).sum()
    test_loss /= len(test_loader.dataset)
    test_loss_lenet.append(test_loss)
    test_accuracy_lenet.append(int(correct))
    print('\nTest set: Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

torch.manual_seed(123)

model = LeNet()

lr = 1*math.exp(-3)
momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_batch_size = 100
test_batch_size = 100
train_loader, test_loader = load_data(train_batch_size, test_batch_size)

epochs = 5
log_interval = 100

start_time_lenet = datetime.now()

print ("\n LeNet-5: \n")

for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader, log_interval=log_interval)
    test(model, epoch, test_loader)

end_time_lenet = datetime.now()
print('Duration of LeNet-5: {}'.format(end_time_lenet - start_time_lenet))

avg_train_loss_lenet = np.mean(train_loss_lenet)
avg_test_loss_lenet = np.mean(test_loss_lenet)
avg_test_accuracy_lenet = round((100 * (np.mean(test_accuracy_lenet)/len(test_loader.dataset))), 2)
max_test_accuracy_lenet = round((100 * (np.max(test_accuracy_lenet)/len(test_loader.dataset))), 2)

print ("Average Training Loss of LeNet-5: " + str(avg_train_loss_lenet))
print ("Average Test Accuracy of LeNet-5: " + str(avg_test_accuracy_lenet))
print ("Maximum Test Accuracy of LeNet-5: " + str(max_test_accuracy_lenet))


###############################################################################
############################ Base CNN #########################################
###############################################################################

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(5 * 5 * 64, 840)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(840, 840)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(840, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        return out

def load_data(train_batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((32,32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=train_batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((32,32)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
    batch_size=test_batch_size, shuffle=True)

    return (train_loader, test_loader)

train_loss_cnn = []
test_loss_cnn = []
test_accuracy_cnn = []

def train(model, optimizer, epoch, train_loader, log_interval):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train set: Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item()))
            train_loss_cnn.append(loss.item())

def test(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += loss_fn(output, target).item()
        pred = np.argmax(output.data, axis=1)
        correct = correct + np.equal(pred, target.data).sum()
    test_loss /= len(test_loader.dataset)
    test_loss_cnn.append(test_loss)
    test_accuracy_cnn.append(int(correct))
    print('\nTest set: Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

torch.manual_seed(123)

model = ConvNet()

lr = 1*math.exp(-3)
momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_batch_size = 100
test_batch_size = 100
train_loader, test_loader = load_data(train_batch_size, test_batch_size)

epochs = 5
log_interval = 100

start_time_cnn = datetime.now()

print ("\n \n CNN: \n")

for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader, log_interval=log_interval)
    test(model, epoch, test_loader)

end_time_cnn = datetime.now()
print('Duration of CNN: {}'.format(end_time_cnn - start_time_cnn))

avg_train_loss_cnn = np.mean(train_loss_cnn)
avg_test_accuracy_cnn = round((100 * (np.mean(test_accuracy_cnn)/len(test_loader.dataset))), 2)
max_test_accuracy_cnn = round((100 * (np.max(test_accuracy_cnn)/len(test_loader.dataset))), 2)

print ("Average Training Loss of CNN: " + str(avg_train_loss_cnn))
print ("Average Test Accuracy of CNN: " + str(avg_test_accuracy_cnn))
print ("Maximum Test Accuracy of CNN: " + str(max_test_accuracy_cnn))


###############################################################################
############################ Plot of Training Loss ############################
###############################################################################

print ("\n \n Plot of Training Loss of LeNet-5 and CNN Model: \n")

plt.plot(train_loss_lenet, color = "blue", label = "LeNet-5")
plt.plot(train_loss_cnn, color = "red", label = "CNN")
plt.ylabel("Training Loss")
plt.legend()
