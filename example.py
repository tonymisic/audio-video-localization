from torch.utils.data.dataloader import DataLoader
from dataloader import FastAVE
import torch, torch.nn as nn, torch.optim as optim
from models import Pepe
from losses import SupConLoss
# globals
DEV, MULTI_GPU = True, False
FEATURE_BANK = 'AVE_Dataset/AVE_Features/'
EPOCHS = 10
LEARNING_RATE = 0.001
LOSS_TEMP = 0.07
BATCH_SIZE = 9
VIDEO_WEIGHT = 0.5
# device 
device = torch.device("cuda")
# training data
train_data = FastAVE(FEATURE_BANK, 'train')
train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True)
# testing data
test_data = FastAVE(FEATURE_BANK, 'test')
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# model
model = Pepe(normalize=True)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
if torch.cuda.device_count() > 1 and MULTI_GPU:
    model = nn.DataParallel(model)
model.to(device)
# pipeline
epoch = 0
while epoch <= EPOCHS:
    ### ------------- TRAINING ------------- ###
    for video, audio in train_loader:
        video = torch.flatten(video, start_dim=2, end_dim=4) 
        video, audio = video.type(torch.FloatTensor).to(device), audio.type(torch.FloatTensor).to(device) # batch x segments x feature
        for segment in range(10):
            video_out, audio_out = model(video[:, segment].squeeze(), audio[:, segment])
            label = torch.zeros([BATCH_SIZE, 10]).to(device)
            label[:, segment] = 1
            vl, al = criterion(video_out, label), criterion(audio_out, label)
            combined_loss = (vl * VIDEO_WEIGHT) + (al * (1 - VIDEO_WEIGHT))
            combined_loss.backward(), optimizer.step()

        if DEV:
            break
    ### ------------- TESTING ------------- ###
    for video, audio, temporal, spatial in test_loader:
        video, audio, temporal, spatial = video.to(device), audio.to(device), temporal.to(device), spatial.to(device)
        video_out, audio_out = model(video, audio)
        # Video 2 Audio

        # Audio 2 Video
        
        if DEV:
            break
    # end of epoch
    epoch += 1
    if DEV:
        break