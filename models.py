from sys import maxsize
from torch import nn
from torch.functional import F
import torch

class AudioModel(nn.Module):
    def __init__(self, input=1280, hidden=512, output=28):
        super(AudioModel, self).__init__()
        self.attention = nn.MultiheadAttention(input, 1)
        self.fc1 = nn.Linear(input, hidden)
        self.classifier = nn.Linear(hidden, output)
    
    def forward(self, x):
        x = self.attention(x,x,x)
        x = F.relu(self.fc1(x))
        return self.classifier(x)

class BinaryClassifier(nn.Module):
    """Model to classify whether audio is background or event
    """
    def __init__(self, input=128, hidden=128, output=1):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.classifier = nn.Linear(hidden, output)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.classifier(x)

class LargeBinaryClassifier(nn.Module):
    """Model to classify whether audio is background or event
    """
    def __init__(self, input=128, hidden=256, output=1):
        super(LargeBinaryClassifier, self).__init__()
        self.layer_1 = nn.Linear(input, hidden)
        self.layer_3 = nn.Linear(hidden, 64)
        self.layer_out = nn.Linear(64, output) 

    def forward(self, x, train=False):
        x = self.layer_1(x)
        x = F.relu(self.layer_3(x))
        if train:
            return self.layer_out(x)
        else:
            return torch.sigmoid(self.layer_out(x))