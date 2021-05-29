import torch, matplotlib.pyplot as plt, torchaudio

def temporal_accuracy(pred, y, threshold):
    assert pred.size() == y.size()
    count, total = 0, 0
    for i in range(pred.size(0)):
        for j in range(pred.size(1)):
            if pred[i][j] > threshold:
                if y[i][j] == 1:
                    count += 1
            else:
                if y[i][j] == 0:
                    count += 1
            total += 1
    return count / total

def load_videos(filename):
    """
    (string) root_dir: root directory of dataset
    (string) filename: .txt file of data annotations
    (list) return: [[class, filename, audio_quality, start, end], ... ]
    """
    data = []
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.rstrip('\n')
        temp = line.split('&')
        data.append(temp)
    return data