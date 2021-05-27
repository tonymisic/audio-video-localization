import torch

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