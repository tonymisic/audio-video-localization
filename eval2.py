from torch.utils.data.dataloader import DataLoader
from audioloader import AVE_Audio
import torch, wandb 
from models import BinaryClassifier
from utils import temporal_accuracy
'''
Main training script
'''
wandb.init(project="Audio Binary Classifier",
    config={
        "task": "Evaluation",
        "dataset": "AVE",
        "device": "GTX1080",
        "batch_size": 1,
        "threshold": 1
    }
)
# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# testing data
val_data = AVE_Audio('AVE_Dataset/', 'val', 'classes.json')
val_loader = DataLoader(val_data, 1, shuffle=True, num_workers=1, pin_memory=True)
# feature extractor
extractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
extractor.eval(), extractor.to(device)
# model
model = BinaryClassifier()
model.load_state_dict(torch.load('models/audio1.pth'))
model.eval(), model.to(device)
### --------------- EVAL --------------- ###
running_accuracy, batch = 0.0, 0
for audio_files, spatial_labels, temporal_labels in val_loader:
    spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
    features = torch.zeros([1, 10, 128])
    for i, v in enumerate(audio_files, 0):
        features[i] = torch.divide(extractor(v), 255)
    features = features.to(device)
    preds = torch.zeros([1, 10])
    for i in range(1):
        for j in range(10):
            preds[i][j] = model(features[i,j,:])
    preds = preds.to(device)
    running_accuracy += float(temporal_accuracy(preds, temporal_labels, 1))
    batch += 1
    wandb.log({"Eval Accuracy": running_accuracy / batch})