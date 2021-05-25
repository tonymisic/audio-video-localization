from MLP import MLP
from torch.utils.data.dataloader import DataLoader
from dataloader import AVE
from torchvideotransforms import video_transforms, volume_transforms
import torchvision.models as models, torch, wandb, torch.optim as optim 
from AttentionModels import SelfAttention, AudioGuidedAttention
'''
Main training script
'''
wandb.init(project="Dual-Attention Matching",
    config={
        "task": "Discriminative",
        "learning_rate": 0.001,
        "dataset": "AVE",
        "device": "GTX1080",
        "epochs": 50,
        "batch_size": 8,
        "record_rate": 10
    }
)
activation = {}
# from https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# transforms
v_transforms = video_transforms.Compose([
    video_transforms.Resize((224, 224), interpolation='nearest'),
    volume_transforms.ClipToTensor()
])
a_transforms = video_transforms.Compose([
    volume_transforms.ClipToTensor()
])
# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# training data
train_data = AVE('AVE_Dataset/', 'train', 'classes.json', v_transforms)
train_loader = DataLoader(train_data, wandb.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
# testing data
test_data = AVE('AVE_Dataset/', 'test', 'classes.json', v_transforms)
test_loader = DataLoader(test_data, wandb.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
# feature extractors
video_model = models.vgg19(pretrained=True)
video_model.avgpool.register_forward_hook(get_activation('pool5'))
video_model.eval(), video_model.to(device)
audio_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
audio_model.eval(), audio_model.to(device)
# models
audio_attention_model = SelfAttention(128)
audio_attention_model.load_state_dict(torch.load('models/audio/guided_epoch11.pth'))
video_attention_model = SelfAttention(128)
video_attention_model.load_state_dict(torch.load('models/video/guided_epoch11.pth'))
guided_model = AudioGuidedAttention()
guided_model.load_state_dict(torch.load('models/guided/guided_epoch11.pth'))
classifier = MLP(256, 28)
classifier.load_state_dict(torch.load('models/classifier/guided_epoch11.pth'))
# losses
criterion = torch.nn.CrossEntropyLoss()
optimizer_classifier = optim.SGD(classifier.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_guided = optim.SGD(guided_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_video = optim.SGD(video_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
optimizer_audio = optim.SGD(audio_attention_model.parameters(), lr=wandb.config['learning_rate'], momentum=0.9)
# send to device
classifier.to(device), video_attention_model.to(device), audio_attention_model.to(device), guided_model.to(device)
criterion.to(device)
# video segmenting
def segment_video_batch(batch):
    segmented_videos = torch.zeros([batch.size(0), 10, 512 * 7 * 7])
    for j, video in enumerate(batch, 0):
        video_features = torch.zeros([video.size(1), 512, 7, 7])
        for i in range(video.size(1)):
            current_frame = video[:, i, :, :]
            current_frame = current_frame.unsqueeze(0)
            _ = video_model(current_frame)
            feature = activation['pool5']
            video_features[i, :, :, :] = feature
        segmented_video, length = torch.zeros([10, 512 * 7 * 7]), video_features.size(0)
        for x, i in enumerate(range(0, length, length // 10), 0):
            segmented_video[x, :] = torch.flatten(torch.mean(video_features[i:i + length // 10, :, :, :], dim=0))
        segmented_videos[j, :, :] = segmented_video
    return segmented_videos

# training loop
epoch = 12
while epoch <= wandb.config['epochs']:
    print("Epoch: " + str(epoch) + " started!")
    running_loss, running_accuracy, batch = 0.0, 0.0, 0
    ### --------------- TRAIN --------------- ###
    for video_batch, audio_files, spatial_labels, temporal_labels in train_loader:
        try:
            optimizer_audio.zero_grad(), optimizer_classifier.zero_grad(), optimizer_video.zero_grad()
            spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
            # segment audio [batch_sample, second, 128]
            segmented_audio = torch.zeros([wandb.config['batch_size'], 10, 128])
            for i, f in enumerate(audio_files, 0):
                segmented_audio[i, :, :] = audio_model(f)
            segmented_audio = torch.div(segmented_audio, 255) # normalize
            video_batch, segmented_audio = video_batch.to(device), segmented_audio.to(device)
            # segment video [batch_sample, second, 25088]
            segmented_video = segment_video_batch(video_batch).to(device)
            # audio-guided attention 
            final_seg_videos = torch.zeros([wandb.config['batch_size'], 10, 128]).to(device)
            for i in range(wandb.config['batch_size']):
                for j in range(10):
                    final_seg_videos[i,j,:] = torch.dot(
                                        guided_model(segmented_video[i,j,:], segmented_audio[i,j,:]), 
                                        segmented_audio[i,j,:]
                                    )
            # self-attention for visual and audio features
            a_atten, a_weight = audio_attention_model(segmented_audio.permute([1,0,2]))
            v_atten, v_weight = video_attention_model(final_seg_videos.permute([1,0,2]))
            # global feature representation
            video_global = torch.mean(v_atten, dim=0)
            audio_global = torch.mean(a_atten, dim=0)
            # event category prediction (my assumption)
            event_relevance = classifier(torch.cat([video_global, audio_global], dim=1))
            predictions, ground_truth = torch.max(event_relevance, 1)[1], torch.max(spatial_labels, 1)[1]
            # calculate loss
            loss = criterion(event_relevance, ground_truth)
            loss.backward()
            optimizer_classifier.step(), optimizer_audio.step(), optimizer_video.step(), optimizer_guided.step()
            running_loss += loss
            running_accuracy += torch.sum(predictions == ground_truth)
            batch += 1
            wandb.log({"Training Loss": running_loss / batch})
            wandb.log({"batch": batch})
        except Exception as e:
            print(str(e))
            print("Caught training error on Epoch: " + str(epoch) + " Batch: " + str(batch))
    wandb.log({"Training Accuracy": (running_accuracy / batch) / wandb.config['batch_size']})
    torch.save(audio_attention_model.state_dict(), 'models/audio/guided_epoch' + str(epoch) + '.pth')
    torch.save(video_attention_model.state_dict(), 'models/video/guided_epoch' + str(epoch) + '.pth')
    torch.save(classifier.state_dict(), 'models/classifier/guided_epoch' + str(epoch) + '.pth')
    torch.save(guided_model.state_dict(), 'models/guided/guided_epoch' + str(epoch) + '.pth')
    print("Saved Models for Epoch:" + str(epoch))
    ### --------------- TEST --------------- ###
    running_loss, running_accuracy, batch = 0.0, 0.0, 0
    for video_batch, audio_files, spatial_labels, temporal_labels in test_loader:
        try:
            spatial_labels, temporal_labels = spatial_labels.to(device), temporal_labels.to(device)
            # segment audio [batch_sample, second, 128]
            segmented_audio = torch.zeros([wandb.config['batch_size'], 10, 128])
            for i, f in enumerate(audio_files, 0):
                segmented_audio[i, :, :] = audio_model(f)
            segmented_audio = torch.div(segmented_audio, 255) # normalize
            video_batch, segmented_audio = video_batch.to(device), segmented_audio.to(device)
            # segment video [batch_sample, second, 25088]
            segmented_video = segment_video_batch(video_batch).to(device)
            # audio-guided attention 
            final_seg_videos = torch.zeros([wandb.config['batch_size'], 10, 128]).to(device)
            for i in range(wandb.config['batch_size']):
                for j in range(10):
                    final_seg_videos[i,j,:] = torch.dot(
                                guided_model(segmented_video[i,j,:], segmented_audio[i,j,:]), 
                                segmented_audio[i,j,:]
                            )
            # self-attention for visual and audio features
            a_atten, a_weight = audio_attention_model(segmented_audio.permute([1,0,2]))
            v_atten, v_weight = video_attention_model(final_seg_videos.permute([1,0,2]))
            # global feature representation (WITH BACKGROUND!)
            video_global = torch.mean(v_atten, dim=0)
            audio_global = torch.mean(a_atten, dim=0)
            # event category prediction (my assumption)
            event_relevance = classifier(torch.cat([video_global, audio_global], dim=1))
            predictions, ground_truth = torch.max(event_relevance, 1)[1], torch.max(spatial_labels, 1)[1]
            running_accuracy += torch.sum(predictions == ground_truth)
            loss = criterion(event_relevance, ground_truth)
            running_loss += loss
            batch += 1
            wandb.log({"Testing Loss": running_loss / batch})
        except Exception as e:
            print(str(e))
            print("Caught testing error on Epoch: " + str(epoch) + " Batch: " + str(batch))
    wandb.log({"Testing Accuracy": (running_accuracy / batch) / wandb.config['batch_size']})
    wandb.log({"epoch": epoch + 1})
    epoch += 1
    print("Epoch: " + str(epoch - 1) + " finished!")
print("Done, done.")