from torch.utils.data import Dataset
import h5py, torch, random
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
        rand = random.sample(range(10), 1)
        video = self.video_features[self.order[index]][rand]
        audio = self.audio_features[self.order[index]][rand]
        temporal_label = self.temporal_labels[self.order[index]][rand]
        # spatial_label = self.spatial_labels[self.order[index]][rand]
        label = torch.zeros([10])
        label[rand] = 1
        if self.split == 'train':
            return video.squeeze(), audio.squeeze(), label
        else:
            return video.squeeze(), audio.squeeze(), label, temporal_label

    def data_from_file(self, file):
        with h5py.File(file, 'r') as hf:
            return hf[list(hf.keys())[0]][:]

    def __len__(self):
        return len(self.order)