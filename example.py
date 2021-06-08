from torch.utils.data.dataloader import DataLoader
from dataloader import FastAVE
import torch
# globals
FEATURE_BANK = 'AVE_Dataset/AVE_Features/'
EPOCHS = 10
DEV = True
# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# training data
train_data = FastAVE(FEATURE_BANK, 'train')
train_loader = DataLoader(train_data, 9, shuffle=True, num_workers=3, pin_memory=True)
# testing data
test_data = FastAVE(FEATURE_BANK, 'test')
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# model

# pipeline
epoch = 0
while epoch <= EPOCHS:
    ### ------------- TRAINING ------------- ###
    for video, audio, shf_video, shf_audio, video_idx, audio_idx in train_loader:
        video, audio, shf_video, shf_audio = video.to(device), audio.to(device), shf_video.to(device), shf_audio.to(device)
        video_idx, audio_idx = video_idx.to(device), audio_idx.to(device)

        if DEV:
            break
    ### ------------- TESTING ------------- ###
    for video, audio, temporal, spatial in test_loader:
        video, audio = torch.from_numpy(video).to(device), torch.from_numpy(audio).to(device)
        temporal, spatial = torch.from_numpy(temporal).to(device), torch.from_numpy(spatial).to(device)

        if DEV:
            break
    # end of epoch
    epoch += 1
    if DEV:
        break