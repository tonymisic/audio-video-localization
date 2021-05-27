from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2, torch, numpy as np, torchaudio, json, random
import torchvision.models as models
class AVE_Audio(Dataset):
    '''
    Custom Dataloader for the Audio-Visual Events Dataset.
    '''
    def __init__(self, root_dir, split, class_json, num_classes=28, num_segments=10):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.info = None
        if split == 'train':
            self.info = self.load_files(root_dir + 'trainSet.txt')
        elif split == 'test':
            self.info = self.load_files(root_dir + 'testSet.txt')
        elif split == 'val':
            self.info = self.load_files(root_dir + 'valSet.txt')
        self.class_map = json.load(open(root_dir + class_json))
    
    def __getitem__(self, index):
        # temporal 10 segment label
        temporal_label = torch.zeros(self.num_segments)
        temporal_label[list(range(int(self.info[index][3]), int(self.info[index][4])))] = 1 
        # content classification label
        spatial_label = torch.zeros(self.num_classes)
        spatial_label[self.class_map[self.info[index][0]]] = 1
        # audio filename
        audio_file = self.root_dir + 'AVE_Audio/' + self.info[index][1] + '.wav'
        return audio_file, spatial_label, temporal_label
        
    def __len__(self):
        return len(self.info)

    def load_files(self, filename):
        """
        (string) root_dir: root directory of dataset
        (string) filename: .txt file of data annotations
        (list) return: [[class, filename, audio_quality, start, end], ... ]
        """
        data = []
        f = open(filename, 'r')
        for line in f.readlines():
            line = line.rstrip('\n')
            temp = line.split('&')
            data.append(temp)
        return data