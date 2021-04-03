import torch
import torch.nn.functional as F
import torch.nn as nn


class ZenNet(nn.Module):
    def __init__(self):
        super(ZenNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, 5)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 5)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*128*128, 64*128)
        self.fc2 = nn.Linear(64*4, 64) 
        self.final = nn.Linear(64, 24) # 24 categories
        self.dropoutConv = nn.Dropout2d(0.25)
        self.dropoutLinear = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropoutConv(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropoutConv(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.dropoutConv(x)
        x = torch.flatten(x,1)
        x = self.dropoutLinear(self.fc1(x))
        x = self.dropoutLinear(self.fc2(x))
        return nn.softmax(self.final(x))
