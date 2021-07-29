import json, h5py, math, random, numpy as np

class ZeroShot():
    ''' Set of functions for spliting AVE into zero-shot learning
    '''
    def __init__(self, save_loc, folder='AVE_Dataset/AVE_Features/', classes_file='AVE_Dataset/classes.json'):
        self.labels = self.data_from_file(folder + 'labels.h5')
        self.temporal = self.data_from_file(folder + 'temporal_labels.h5')
        self.train = self.data_from_file(folder + 'train_order.h5')
        self.test = self.data_from_file(folder + 'test_order.h5')
        self.val = self.data_from_file(folder + 'val_order.h5')
        self.class_map = json.load(open(classes_file))
        self.length = len(self.class_map)
        self.inclusion_list = list(range(29))
        self.save = save_loc

    def data_from_file(self, file):
        with h5py.File(file, 'r') as hf:
            return hf[list(hf.keys())[0]][:]

    def split_data(self, pos_split=None, neg_classes=None):
        assert pos_split != None or neg_classes != None
        if pos_split != None:
            if neg_classes == None:
                self.split_by_percent(pos_split)
        elif pos_split == None:
            if neg_classes != None:
                self.split_by_classes(neg_classes)

    def split_by_percent(self, percent):
        assert percent < 1.0 and percent > 0.0
        print("Split by percentage positives: " + str(round(percent * 100)) + "%")
        positives, pos_count = math.floor(percent * self.length), 0
        for i in random.sample(range(0, self.length), self.length):
            if pos_count < positives:
                self.inclusion_list[i] = 1
                pos_count += 1
            else:
                self.inclusion_list[i] = 0
        self.split()
        self.print_classes()

    def split_by_classes(self, classes):
        assert len(classes) < self.length
        print("Split by classes: " + str(classes))
        for i in range(self.length):
            if i in classes:
                self.inclusion_list[i] = 0
            else:
                self.inclusion_list[i] = 1
        self.split()
        self.print_classes()

    def split(self):
        training, testing = [], []
        for i, val in enumerate(self.labels, 0):
            if self.inclusion_list[np.argmax(val)] == 1:
                training.append(i)
            else:
                if np.sum(self.temporal[i]) < 10:
                    testing.append(i)
        hf1, hf2 = h5py.File(self.save + 'trainingZSL.h5', 'w'), h5py.File(self.save + 'testingZSL.h5', 'w')
        hf1.create_dataset('dataset', data=np.array(training)), hf2.create_dataset('dataset', data=np.array(testing))
        hf1.close(), hf2.close() 

    def print_classes(self):
        pos_list, neg_list = [], []
        for name, value in self.class_map.items():
            if self.inclusion_list[value] == 1:
                pos_list.append(name)
            else:
                neg_list.append(name)
        print("------------------------------------------------")
        print("Classes in training: " + str(pos_list))
        print("------------------------------------------------")
        print("Classes in validation: " + str(neg_list))
        print("------------------------------------------------")
        print("Split complete, saved in: " + self.save)


print("")