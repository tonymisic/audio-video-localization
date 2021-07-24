import torch, torchaudio, numpy as np, matplotlib.pyplot as plt
from utils import load_videos
from PIL import Image

loc = 'AVE_Dataset/Annotations.txt'
info = load_videos(loc)

for i, v in enumerate(info, 1):
    if str(info[i][3]) != 0 and str(info[i][4]) != 10:
        print("Event Info: " + str(info[i][3]) + " - " + str(info[i][4]))
        audio_file = 'AVE_Dataset/AVE_Audio/' + info[i][1] + '.wav'
        waveform, sf = torchaudio.load(audio_file)
        plt.figure()
        plt.plot(waveform[0])
        plt.savefig('audio1.png')
        plt.figure()
        plt.plot(waveform[1])
        plt.savefig('audio2.png')
        input("Enter to continue")