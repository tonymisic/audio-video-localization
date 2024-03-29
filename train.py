from torch.utils.data.dataloader import DataLoader
from dataloader import FastAVE
import torch, wandb, torch.optim as optim 
from transformers import AutoTokenizer, AutoModel
from models import Video, Audio, Text
from splitter import ZeroShot
from utils import video_accuracy, class_accuracy
'''
Main training script
'''
wandb.init(project="Zero Shot Learning",
    config={
        "task": "Classification",
        "learning_rate": 0.001,
        "dataset": "AVE",
        "device": "GTX1080",
        "epochs": 999,
        "starting_epoch" : 0,
        "batch_size": 21,
        "threshold": 0.5,
        "eval_classes": [0,1,2,3,4]
    }
)
# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seperate classes
zsl = ZeroShot('AVE_Dataset/ZSL_Features/')
zsl.split_data(neg_classes=[0,1,2,3,4])
# training data
train_data = FastAVE('AVE_Dataset/AVE_Features/', 'train', ZSL=True)
train_loader = DataLoader(train_data, wandb.config['batch_size'], shuffle=True, num_workers=3, pin_memory=True)
# testing data
test_data = FastAVE('AVE_Dataset/AVE_Features/', 'val', ZSL=True)
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# feature extractor
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
extractor = AutoModel.from_pretrained("bert-base-uncased")
extractor.eval(), extractor.to(device)
# model
video_model, audio_model, text_model = Video(out=1, normalize=True), Audio(out=1, normalize=True), Text(out=29)
criterion_video = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.18]).to(device))
criterion_audio = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.18]).to(device))
opt_video = optim.SGD(video_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
opt_audio = optim.SGD(audio_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
opt_text = optim.SGD(text_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
video_model.to(device), audio_model.to(device), text_model.to(device)
# training loop
epoch = wandb.config['starting_epoch']
while epoch <= wandb.config['epochs']:
    print("Epoch: " + str(epoch) + " started!")
    running_lossV, running_accuracyV, batch = 0.0, 0.0, 0
    running_lossA, running_accuracyA = 0.0, 0.0
    ### --------------- TRAIN --------------- ###
    for video, audio, temporal_labels, spatial_labels, class_names in train_loader:
        video, audio = torch.flatten(video, start_dim=2).to(device), audio.to(device)
        temporal_labels, spatial_labels = temporal_labels.to(device), spatial_labels.to(device)
        opt_video.zero_grad(), opt_audio.zero_grad()
        embeddings = torch.zeros([video.size(0), video.size(1), 768]).to(device)
        for i in range(video.size(0)):
            for j in range(10):
                inputs = tokenizer(class_names[j][i], return_tensors="pt")
                inputs = inputs.to(device)
                embeddings[i][j] = extractor(**inputs)['pooler_output'][0]    
        video_pred = video_model(video).squeeze(dim=2)
        audio_pred = audio_model(audio).squeeze(dim=2)
        text_pred = text_model(embeddings)
        video_loss = criterion_video(video_pred, temporal_labels)
        video_loss.backward(), opt_video.step()
        audio_loss = criterion_audio(audio_pred, temporal_labels)
        audio_loss.backward(), opt_audio.step()
        running_lossV += video_loss
        running_lossA += audio_loss
        running_accuracyV += video_accuracy(video_pred, temporal_labels, wandb.config['threshold'], device)
        running_accuracyA += video_accuracy(audio_pred, temporal_labels, wandb.config['threshold'], device)
        batch += 1
        wandb.log({"batch": batch})
        wandb.log({"Video Loss": running_lossV / batch})
        wandb.log({"Audio Loss": running_lossA / batch})
    wandb.log({"Training Video Accuracy": running_accuracyV / batch})
    wandb.log({"Training Audio Accuracy": running_accuracyA / batch})
    torch.save(audio_model.state_dict(), 'models/audio/audio' + str(epoch) + '.pth')
    torch.save(video_model.state_dict(), 'models/video/video' + str(epoch) + '.pth')
    print("Saved Models for Epoch:" + str(epoch))
    ### --------------- TEST --------------- ###
    running_lossV, running_accuracyV, batch = 0.0, 0.0, 0
    running_lossA, running_accuracyA = 0.0, 0.0
    for video, audio, temporal_labels, spatial_labels, class_names in test_loader:
        video, audio = torch.flatten(video, start_dim=2).to(device), audio.to(device)
        temporal_labels, spatial_labels = temporal_labels.to(device), spatial_labels.to(device)
        embeddings = torch.zeros([video.size(0), video.size(1), 768]).to(device)
        for i in range(video.size(0)):
            for j in range(10):
                inputs = tokenizer(class_names[j][i], return_tensors="pt")
                inputs = inputs.to(device)
                embeddings[i][j] = extractor(**inputs)['pooler_output'][0]    
        video_pred = video_model(video).squeeze(dim=2)
        audio_pred = audio_model(audio).squeeze(dim=2)
        text_pred = text_model(embeddings)
        running_accuracyV += video_accuracy(video_pred, temporal_labels, wandb.config['threshold'], device)
        running_accuracyA += video_accuracy(audio_pred, temporal_labels, wandb.config['threshold'], device)
        batch += 1
    wandb.log({"ZeroShot Video Accuracy": running_accuracyV / batch})
    wandb.log({"ZeroShot Audio Accuracy": running_accuracyA / batch})
    wandb.log({"epoch": epoch + 1})
    epoch += 1
    print("Epoch: " + str(epoch - 1) + " finished!")
print("Done, done, and done.")