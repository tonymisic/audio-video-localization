from torch.utils.data import Dataset
import h5py, torch, random, json
class FastAVE(Dataset):
    '''
    Precomputed Feature Dataloader for the Audio-Visual Events Dataset.
    '''
    def __init__(self, root_dir, split, class_json='AVE_Dataset/classes.json'):
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
        self.class_map = json.load(open(class_json))
        
    def __getitem__(self, index):
        video = torch.from_numpy(self.video_features[self.order[index]]).type(torch.FloatTensor)
        audio = torch.from_numpy(self.audio_features[self.order[index]]).type(torch.FloatTensor)
        temporal_label = self.temporal_labels[self.order[index]]
        spatial_label = self.spatial_labels[self.order[index]]
        class_names = self.get_class_names(spatial_label)
        if self.split == 'train':
            return video.squeeze(), audio.squeeze(), temporal_label, class_names
        else:
            return video.squeeze(), audio.squeeze(), temporal_label, spatial_label, class_names

    def __len__(self):
        return len(self.order)
    
    def data_from_file(self, file):
        with h5py.File(file, 'r') as hf:
            return hf[list(hf.keys())[0]][:]

    def get_class_names(self, spatial_labels):
        classes = []
        for i in torch.from_numpy(spatial_labels):
            for name, value in self.class_map.items():
                if value == torch.argmax(i, dim=0):
                    classes.append(name)
                    break
        return classes
