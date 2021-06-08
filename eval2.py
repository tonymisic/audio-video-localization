from torch.utils.data.dataloader import DataLoader
from audioloader import AVE_Audio
import torch, wandb 
from models import LargeBinaryClassifier
from utils import temporal_accuracy, class_accuracy
'''
Main training script
'''
wandb.init(project="Audio Binary Classifier",
    config={
        "task": "Evaluation",
        "dataset": "AVE",
        "device": "GTX1080",
        "batch_size": 1,
        "threshold": 0.5
    }
)
# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# testing data
val_data = AVE_Audio('AVE_Dataset/', 'val', 'classes.json')
val_loader = DataLoader(val_data, wandb.config['batch_size'], shuffle=True, num_workers=1, pin_memory=True)
# feature extractor
extractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
extractor.eval(), extractor.to(device)
# model
model = LargeBinaryClassifier()
model.load_state_dict(torch.load('models/audio10.pth'))
model.eval(), model.to(device)
### --------------- EVAL --------------- ###
running_accuracy, batch = 0.0, 0
background_acc, event_acc = 0, 0
for audio_files, spatial_labels, temporal_labels in val_loader:
    spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
    features = torch.zeros([wandb.config['batch_size'], 10, 128])
    for i, v in enumerate(audio_files, 0):
        features[i] = torch.divide(extractor(v), 255)
    features = features.to(device)
    preds = torch.zeros([wandb.config['batch_size'], 10])
    for i in range(wandb.config['batch_size']):
        for j in range(10):
            preds[i][j] = model(features[i,j,:])
    preds = preds.to(device)
    running_accuracy += float(temporal_accuracy(preds, temporal_labels, wandb.config['threshold']))
    # class statistics
    c1, c2 = class_accuracy(preds, temporal_labels, wandb.config['threshold'])
    background_acc += float(c1)
    event_acc += float(c2)
    batch += 1
    wandb.log({"Eval Accuracy": running_accuracy / batch})
    wandb.log({"Eval Event Accuracy": event_acc / batch})
    wandb.log({"Eval Background Accuracy": background_acc / batch})