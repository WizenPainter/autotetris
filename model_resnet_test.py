import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet(nn.Module):
  def __init__(self, num_classes=200):
    super(ResNet, self).__init__()
    
    self.model_name='resnet_jaime'
    self.model=models.resnet50()

    self.model.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.model.bn1 = nn.BatchNorm2d(32)
    self.model.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.model.bn2 = nn.BatchNorm2d(32)
    self.model.conv3 = nn.Conv2d(32, 32, 3, padding=1)
    self.model.bn3 = nn.BatchNorm2d(32)
    self.model.pool1 = nn.MaxPool2d(2, 2)
    self.model.conv4 = nn.Conv2d(32, 64, 3, padding=1)
    self.model.bn4 = nn.BatchNorm2d(64)
    self.model.conv5 = nn.Conv2d(64, 64, 3, padding=1)
    self.model.bn5 = nn.BatchNorm2d(64)
    self.model.conv6 = nn.Conv2d(64, 32, 3, padding=1)
    self.model.bn6 = nn.BatchNorm2d(32)
    self.model.pool2 = nn.MaxPool2d(2, 2)
    self.model.conv7 = nn.Conv2d(32, 16, 3, padding=1)
    self.model.bn7 = nn.BatchNorm2d(16)
    self.model.conv8 = nn.Conv2d(16, 8, 3, padding=1)
    self.model.bn8 = nn.BatchNorm2d(8)
    self.model.conv9 = nn.Conv2d(8, 2, 3, padding=1)
    self.model.bn9 = nn.BatchNorm2d(2)
    self.model.pool3 = nn.MaxPool2d(2, 2)

    self.model.fc1 = nn.Linear(128 * 4 * 4, 1024)
    self.model.fc2 = nn.Linear(1024, 256)
    self.model.fc3 = nn.Linear(256, 64)
    self.model.fc4 = nn.Linear(64, 32)
    self.model.fc5 = nn.Linear(32, num_classes)

  def forward(self, x):
    x = F.relu(self.model.bn1(self.model.conv1(x)))
    x = F.relu(self.model.bn2(self.model.conv2(x)))
    x = F.relu(self.model.bn3(self.model.conv3(x)))
    x = self.model.pool1(x)
    x = F.relu(self.model.bn4(self.model.conv4(x)))
    x = F.relu(self.model.bn5(self.model.conv5(x)))
    x = F.relu(self.model.bn6(self.model.conv6(x)))
    x = self.model.pool2(x)
    x = F.relu(self.model.bn7(self.model.conv7(x)))
    x = F.relu(self.model.bn8(self.model.conv8(x)))
    x = F.relu(self.model.bn9(self.model.conv9(x)))
    x = self.model.pool3(x)
    x = x.view(-1, 128 * 4 * 4)
    x = F.relu(self.model.fc1(x))
    x = F.relu(self.model.fc2(x))
    x = F.relu(self.model.fc3(x))
    x = F.relu(self.model.fc4(x))
    x = self.model.fc5(x)
    return x