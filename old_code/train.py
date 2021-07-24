from torch.utils.data.dataloader import DataLoader
from audioloader import AVE_Audio
import torch, wandb, torch.optim as optim 
from models import LargeBinaryClassifier
from utils import temporal_accuracy, class_accuracy, video_accuracy
'''
Main training script
'''
wandb.init(project="Audio Binary Classifier",
    config={
        "task": "Classification",
        "learning_rate": 0.001,
        "dataset": "AVE",
        "device": "GTX1080",
        "epochs": 60,
        "batch_size": 21,
        "threshold": 0.5,
        "bckgrnd_weight": 0.82,
        "event_weight": 0.18,
    }
)
# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# training data
train_data = AVE_Audio('AVE_Dataset/', 'train', 'classes.json')
train_loader = DataLoader(train_data, wandb.config['batch_size'], shuffle=True, num_workers=3, pin_memory=True)
# testing data
test_data = AVE_Audio('AVE_Dataset/', 'test', 'classes.json')
test_loader = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# feature extractor
extractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
extractor.eval(), extractor.to(device)
# model
model = LargeBinaryClassifier()
model.load_state_dict(torch.load('models/audio30.pth'))
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.18]).to(device))
optimizer = optim.SGD(model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
model.to(device)
# training loop
epoch = 31
while epoch <= wandb.config['epochs']:
    print("Epoch: " + str(epoch) + " started!")
    running_loss, running_accuracy, batch = 0.0, 0.0, 0
    background_acc, event_acc, video_acc = 0, 0, 0
    ### --------------- TRAIN --------------- ###
    for audio_files, spatial_labels, temporal_labels in train_loader:
        spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
        optimizer.zero_grad()
        features = torch.zeros([wandb.config['batch_size'], 10, 128])
        for i, v in enumerate(audio_files, 0):
            features[i] = torch.divide(extractor(v), 255)
        features = features.to(device)
        preds = torch.zeros([wandb.config['batch_size'], 10])
        for i in range(wandb.config['batch_size']):
            for j in range(10):
                preds[i][j] = model(features[i,j,:], train=True)
        preds = preds.to(device)
        loss = criterion(preds, temporal_labels)
        loss.backward(), optimizer.step()
        running_loss += float(loss)
        running_accuracy += float(temporal_accuracy(torch.sigmoid(preds), temporal_labels, wandb.config['threshold']))
        video_acc += float(video_accuracy(torch.sigmoid(preds), temporal_labels, wandb.config['threshold'], device))
        c1, c2 = class_accuracy(torch.sigmoid(preds), temporal_labels, wandb.config['threshold'])
        background_acc += float(c1)
        event_acc += float(c2)
        batch += 1
        wandb.log({"batch": batch})
        wandb.log({"Training Loss": running_loss / batch})
    wandb.log({"Training Accuracy": running_accuracy / batch})
    wandb.log({"Training Event Accuracy": event_acc / batch})
    wandb.log({"Training Background Accuracy": background_acc / batch})
    wandb.log({"Training Video Accuracy": video_acc / batch})
    torch.save(model.state_dict(), 'models/audio' + str(epoch) + '.pth')
    print("Saved Models for Epoch:" + str(epoch))
    ### --------------- TEST --------------- ###
    running_loss, running_accuracy, batch = 0.0, 0.0, 0
    background_acc, event_acc, video_acc = 0, 0, 0
    for audio_files, spatial_labels, temporal_labels in test_loader:
        spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
        optimizer.zero_grad()
        features = torch.zeros([1, 10, 128])
        for i, v in enumerate(audio_files, 0):
            features[i] = torch.divide(extractor(v), 255)
        features = features.to(device)
        preds = torch.zeros([1, 10])
        for i in range(1):
            for j in range(10):
                preds[i][j] = model(features[i,j,:])
        preds = preds.to(device)
        running_loss += float(criterion(preds, temporal_labels))
        running_accuracy += float(temporal_accuracy(preds, temporal_labels, wandb.config['threshold']))
        video_acc += float(video_accuracy(preds, temporal_labels, wandb.config['threshold'], device))
        # class statistics
        c1, c2 = class_accuracy(preds, temporal_labels, wandb.config['threshold'])
        background_acc += float(c1)
        event_acc += float(c2)
        batch += 1
        wandb.log({"Testing Loss": running_loss / batch})
    wandb.log({"Testing Accuracy": running_accuracy / batch})
    wandb.log({"Testing Event Accuracy": event_acc / batch})
    wandb.log({"Testing Background Accuracy": background_acc / batch})
    wandb.log({"Testing Video Accuracy": video_acc / batch})
    wandb.log({"epoch": epoch + 1})
    epoch += 1
    print("Epoch: " + str(epoch - 1) + " finished!")
print("Done, done.")