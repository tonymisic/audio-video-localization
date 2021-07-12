from torch.utils.data.dataloader import DataLoader
from dataloader import FastAVE
import torch, wandb
from models import Pepe
# globals
# wandb.init(project="Unsupervised AVE",
#     config={
#         "DEV": False,
#         "MULTI_GPU" : False,
#         "FEATURE_BANK" : 'AVE_Dataset/AVE_Features/',
#     }
# )
# device 
device = torch.device("cuda")
# val data
val_data = FastAVE('AVE_Dataset/AVE_Features/', 'val')
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# model
model = Pepe(normalize=True)
model.load_state_dict(torch.load('models/unsupervised/unsupervised2.pth'))
model.eval()
model.to(device)
accuracy_video, accuracy_audio = 0, 0
iteration, batch = 0, 0
### ------------- EVALUATION ------------- ###
for video, audio, temporal, spatial in val_loader:
    video = torch.flatten(video, start_dim=2, end_dim=4)
    video, audio, temporal, spatial = video.to(device), audio.to(device), temporal.to(device), spatial.to(device)
    video, audio = video.type(torch.FloatTensor).to(device), audio.type(torch.FloatTensor).to(device)
    for segment in range(10):
        video_out, audio_out = model(video[:, segment].squeeze(dim=1), audio[:, segment])
        label = torch.zeros([1, 10]).to(device)
        label[:, segment] = 1
        video_pred, audio_pred = torch.max(video_out, dim=1), torch.max(audio_out, dim=1)
        label = torch.max(label, dim=1)
        accuracy_video += torch.sum(label.indices == video_pred.indices)
        accuracy_audio += torch.sum(label.indices == audio_pred.indices)
        # write results
        # wandb.log({
        #     "Test Acc Video": accuracy_video,
        #     "Test Acc Audio": accuracy_audio,
        # })
        iteration += 1