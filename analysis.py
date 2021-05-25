import json

def zeros(l):
    array = []
    for i in range(l):
        array.append(0.0)
    return array

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

info_all = load_videos('AVE_Dataset/Annotations.txt')[1:]
info_train = load_videos('AVE_Dataset/trainSet.txt')
info_test = load_videos('AVE_Dataset/testSet.txt')
info_val = load_videos('AVE_Dataset/valSet.txt')
class_map = json.load(open('AVE_Dataset/classes.json'))

for i in class_map:
    class_map[i] = zeros(11)

count, total = 0, 0
for val in info_all:
    start, end, label = int(val[3]), int(val[4]), val[0]
    length = end - start
    class_map[label][length-1] += 1
    class_map[label][10] += 1
for i in class_map:
    freq = zeros(10)
    for j in range(10):
        freq[j] = round((class_map[i][j] / class_map[i][10]) * 100, 2)
    print('Class: {:<30} Frequency: {:<30} Total: {:<10}'.format(i, str(freq), freq.index(max(freq)) + 1))