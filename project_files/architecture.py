import torch
import torch.nn as nn
import numpy as np

class MyCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(MyCNN, self).__init__()
        
        #self.conv1d = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.b_n1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.b_n2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b_n3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.b_n4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*6*6, 1024)
        self.b_n5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.b_n6 = nn.BatchNorm1d(512)

        #self.dropout = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout(p=0.8)

        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        #self.leaky_relu = nn.LeakyReLU()
        #self.ELU = nn.ELU()

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        #x = self.pool(self.relu(self.b_n1(self.conv1d(input_images))))
        x = self.pool(self.relu(self.b_n1(self.conv1(input_images))))
        x = self.pool(self.relu(self.b_n2(self.conv2(x))))
        x = self.pool(self.relu(self.b_n3(self.conv3(x))))
        x = self.pool(self.relu(self.b_n4(self.conv4(x))))
        
        x = x.reshape(-1, 512*6*6)

        x = self.relu(self.b_n5(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.b_n6(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

model = MyCNN(num_classes=20)