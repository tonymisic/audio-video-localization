from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2, torch, numpy as np, torchaudio, json, random
import torchvision.models as models
class AVE(Dataset):
    '''
    Custom Dataloader for the Audio-Visual Events Dataset.
    '''
    def __init__(self, root_dir, split, class_json, video_transform, background=True, num_classes=28, num_segments=10):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.video_transform = video_transform
        self.target_frames = 160
        self.info = None
        self.background = background
        if split == 'train':
            self.info = self.load_videos(root_dir + 'trainSet.txt')
        elif split == 'test':
            self.info = self.load_videos(root_dir + 'testSet.txt')
        elif split == 'val':
            self.info = self.load_videos(root_dir + 'valSet.txt')
        self.class_map = json.load(open(root_dir + class_json))
    
    def __getitem__(self, index):
        # raw video and audio data
        video = self.get_video(self.root_dir + 'AVE/' + self.info[index][1] + '.mp4')
        # temporal 10 segment label
        temporal_label = torch.zeros(self.num_segments)
        temporal_label[list(range(int(self.info[index][3]), int(self.info[index][4])))] = 1 
        # content classification label
        spatial_label = torch.zeros(self.num_classes)
        spatial_label[self.class_map[self.info[index][0]]] = 1
        # apply transforms
        augmented_video = self.video_transform(video)
        audio_file = self.root_dir + 'AVE_Audio/' + self.info[index][1] + '.wav'
        if self.background:
            return augmented_video, audio_file, spatial_label, temporal_label
        else:
            return augmented_video, audio_file, spatial_label, temporal_label, int(self.info[index][3]), int(self.info[index][4])

    def __len__(self):
        return len(self.info)

    def load_videos(self, filename):
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

    def get_video(self, filename):
        """
        (string) filename: .mp4 file
        (FloatTensor) return: video frame data
        """
        cap = cv2.VideoCapture(filename)
        frames = []
        success, image = cap.read()
        while success:
            frames.append(image)
            success, image = cap.read()
        indicies = sorted(random.sample(range(len(frames)), self.target_frames))
        return np.asarray([frames[i] for i in indicies])