import torch
import torch.nn as nn
from .base import MyClassifier

class LinearClassifier(MyClassifier):
    def __init__(self, prior, dim):
        super(LinearClassifier, self).__init__(prior)
        self.input_dim = dim
        if dim == 1:
            self.fc = nn.Linear(784, 1)
        else:
            self.fc = nn.Linear(dim, 1)
        
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        return torch.tanh(self.fc(x)) 