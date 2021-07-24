import torch, random

def video_accuracy(pred, y, threshold, device):
    assert pred.size() == y.size()
    count, total = 0, 0
    for i in range(pred.size(0)):
        current_pred = torch.zeros(pred.size(1)).to(device)
        for j in range(pred.size(1)):
            if pred[i][j] > threshold:
                current_pred[j] = 1
        if torch.equal(current_pred, y[i]):
            count += 1
        total += 1
    return count / total

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

def class_accuracy(pred, y, threshold):
    assert pred.size() == y.size()
    c_0, c_1, t_0, t_1 = 0, 0, 0, 0
    for i in range(pred.size(0)):
        for j in range(pred.size(1)):
            if y[i][j] == 1:
                t_1 += 1
            else:
                t_0 += 1
            if pred[i][j] > threshold:
                if y[i][j] == 1:
                    c_1 += 1
            else:
                if y[i][j] == 0:
                    c_0 += 1
    if (t_0 == 0):
        return 0, c_1 / t_1
    elif (t_1 == 0):
        return c_0 / t_0, 0
    else:
        return c_0 / t_0, c_1 / t_1

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