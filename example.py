from torch.utils.data.dataloader import DataLoader
from dataloader import FastAVE
import torch, torch.nn as nn, torch.optim as optim, wandb, random
from models import Pepe
# globals
wandb.init(project="Unsupervised AVE",
    config={
        "DEV": False,
        "MULTI_GPU" : False,
        "FEATURE_BANK" : 'AVE_Dataset/AVE_Features/',
        "EPOCHS" : 1000,
        "LEARNING_RATE" : 0.001,
        "BATCH_SIZE" : 3339,
        "VIDEO_WEIGHT" : 0.5
    }
)
# device 
device = torch.device("cuda")
# training data
train_data = FastAVE(wandb.config['FEATURE_BANK'], 'train')
train_loader = DataLoader(train_data, wandb.config['BATCH_SIZE'], shuffle=True, num_workers=3, pin_memory=True)
# testing data
test_data = FastAVE(wandb.config['FEATURE_BANK'], 'test')
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# model
model = Pepe(normalize=True)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=wandb.config['LEARNING_RATE'], momentum=0.9)
if torch.cuda.device_count() > 1 and wandb.config['MULTI_GPU']:
    model = nn.DataParallel(model)
model.to(device)
wandb.watch(model, criterion=criterion)
# pipeline
epoch = 0
while epoch <= wandb.config['EPOCHS']:
    wandb.log({"epoch": epoch})
    ### ------------- TRAINING ------------- ###
    accuracy_video, accuracy_audio, loss, loss_a, loss_v = 0, 0, 0, 0, 0
    iteration, batch = 0, 0
    for video, audio, labels in train_loader:
        video = torch.flatten(video, start_dim=1, end_dim=3) 
        labels = labels.to(device)
        video, audio = video.type(torch.FloatTensor).to(device), audio.type(torch.FloatTensor).to(device)
        video_out, audio_out = model(video, audio)
        vl, al = criterion(video_out, labels), criterion(audio_out, labels)
        loss_v += vl
        loss_a += al
        combined_loss = (vl * wandb.config['VIDEO_WEIGHT']) + (al * (1 - wandb.config['VIDEO_WEIGHT']))
        combined_loss.backward(), optimizer.step()
        loss += combined_loss
        video_pred, audio_pred = torch.max(video_out, dim=1), torch.max(audio_out, dim=1)
        labels = torch.max(labels, dim=1)
        accuracy_video += torch.sum(labels.indices == video_pred.indices) 
        accuracy_audio += torch.sum(labels.indices == audio_pred.indices)
        wandb.log({"train batch": batch})
        batch += 1
        # write losses
        wandb.log({
            "Train Loss Video": loss_a / batch,
            "Train Loss Audio": loss_v / batch,
            "Train Loss Both": loss / batch,
        })
        if wandb.config['DEV']:
            break
    # write results
    wandb.log({
        "Train Acc Video": accuracy_video / (wandb.config['BATCH_SIZE'] * batch),
        "Train Acc Audio": accuracy_audio / (wandb.config['BATCH_SIZE'] * batch),
    })
    torch.save(model.state_dict(), 'models/unsupervised/unsupervised' + str(epoch) + '.pth')
    print("Saved Models for Epoch:" + str(epoch))
    accuracy_video, accuracy_audio, loss, loss_a, loss_v = 0, 0, 0, 0, 0
    iteration, batch = 0, 0
    ### ------------- TESTING ------------- ###
    for video, audio, labels, temporal in test_loader:
        video = torch.flatten(video, start_dim=1, end_dim=3) 
        labels = labels.to(device)
        video, audio = video.type(torch.FloatTensor).to(device), audio.type(torch.FloatTensor).to(device)
        video_out, audio_out = model(video, audio)
        vl, al = criterion(video_out, labels), criterion(audio_out, labels)
        loss_v += vl
        loss_a += al
        combined_loss = (vl * wandb.config['VIDEO_WEIGHT']) + (al * (1 - wandb.config['VIDEO_WEIGHT']))
        loss += combined_loss
        video_pred, audio_pred = torch.max(video_out, dim=1), torch.max(audio_out, dim=1)
        labels = torch.max(labels, dim=1)
        accuracy_video += torch.sum(labels.indices == video_pred.indices) 
        accuracy_audio += torch.sum(labels.indices == audio_pred.indices)
        wandb.log({"test batch": batch})
        batch += 1
        # write losses
        wandb.log({
            "Test Loss Video": loss_a / batch,
            "Test Loss Audio": loss_v / batch,
            "Test Loss Both": loss / batch,
        })
        if wandb.config['DEV']:
            break
    # write results
    wandb.log({
        "Test Acc Video": accuracy_video / batch,
        "Test Acc Audio": accuracy_audio / batch,
    })
    print("Epoch: " + str(epoch) + " finished!")
    # end of epoch
    epoch += 1
    if wandb.config['DEV']:
        break