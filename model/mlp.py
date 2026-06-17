import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import MyClassifier

class ThreeLayerPerceptron(MyClassifier):
    def __init__(self, prior, dim):
        super(ThreeLayerPerceptron, self).__init__(prior)
        self.input_dim = dim
        if dim == 1:
            self.fc1 = nn.Linear(784, 100)
        else:
            self.fc1 = nn.Linear(dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h

class MultiLayerPerceptron(MyClassifier):
    def __init__(self, prior, dim):
        super(MultiLayerPerceptron, self).__init__(prior)
        self.input_dim = dim
        input_dim = 784 if dim == 1 else dim
            
        self.fc1 = nn.Linear(input_dim, 300, bias=False)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, 300, bias=False)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, 300, bias=False)
        self.bn3 = nn.BatchNorm1d(300)
        self.fc4 = nn.Linear(300, 300, bias=False)
        self.bn4 = nn.BatchNorm1d(300)
        self.fc5 = nn.Linear(300, 1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.relu(self.bn3(self.fc3(h)))
        h = F.relu(self.bn4(self.fc4(h)))
        h = self.fc5(h)
        return h 