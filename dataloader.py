from torch.utils.data import Dataset
import numpy as np, random, h5py
class FastAVE(Dataset):
    '''
    Precomputed Feature Dataloader for the Audio-Visual Events Dataset.
    '''
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split
        self.spatial_labels = self.data_from_file(root_dir + 'labels.h5')
        self.temporal_labels = self.data_from_file(root_dir + 'temporal_labels.h5')
        self.audio_features = self.data_from_file(root_dir + 'audio_feature.h5')
        self.video_features = self.data_from_file(root_dir + 'visual_feature.h5')
        if split == 'train':
            self.order = self.data_from_file(root_dir + 'train_order.h5')
        elif split == 'test':
            self.order = self.data_from_file(root_dir + 'test_order.h5')
        elif split == 'val':
            self.order = self.data_from_file(root_dir + 'val_order.h5')
        
    def __getitem__(self, index):
        video = self.video_features[self.order[index]]
        audio = self.audio_features[self.order[index]]
        temporal_label = self.temporal_labels[self.order[index]]
        spatial_label = self.spatial_labels[self.order[index]]
        if self.split == 'train':
            s_video, video_idx = self.shuffle(video)
            s_audio, audio_idx = self.shuffle(audio)
            return video, audio, s_video, s_audio, video_idx, audio_idx
        else:
            return video, audio, temporal_label, spatial_label

    def shuffle(self, tensor):
        indicies = random.sample(range(10), 10)
        return tensor[indicies], indicies

    def data_from_file(self, file):
        with h5py.File(file, 'r') as hf:
            return hf[list(hf.keys())[0]][:]

    def __len__(self):
        return len(self.order)