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

class Pepe(nn.Module):
    """Model to predict segment temporal alignment
    """
    def __init__(self, video_size=25088, audio_size=128, segments=10, normalize=False):
        super(Pepe, self).__init__()
        self.normalize = normalize;
        self.video_layers = nn.Sequential(
            nn.Linear(video_size, 4096),
            nn.Linear(4096, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, segments), nn.ReLU()
        )
        self.audio_layers = nn.Sequential(
            nn.Linear(audio_size, audio_size),
            nn.Linear(audio_size, segments), nn.ReLU()
        )
    
    def forward(self, video, audio):
        video = self.video_layers(video)
        audio = self.audio_layers(audio)
        if self.normalize:
            return torch.sigmoid(video), torch.sigmoid(audio)
        else:
            return video, audio


class AudioMoment(nn.Module):
    """Model to predict segment temporal alignment
    """
    def __init__(self, video_size=25088, embed=128, heads=1, output=10):
        super(AudioMoment, self).__init__()
        self.layer_1 = nn.Linear(video_size, 512)
        self.layer_2 = nn.Linear(512, embed)
        self.attention = nn.MultiheadAttention(embed, heads)
        self.classifier = nn.Linear(embed, output)
    
    def forward(self, query, video): # query is audio segment
        assert query.size(0) == 1
        assert video.size(0) == 10
        video = self.layer_1(video) # batch x 10 x 25088
        video = F.relu(self.layer_2(video))
        query, video = self.reorder(query), self.reorder(video)
        x = self.attention(query, video, video)
        x = F.relu(self.classifier(x))
        return torch.sigmoid(x)