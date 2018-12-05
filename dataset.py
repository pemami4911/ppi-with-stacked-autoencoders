import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PPIDataset(Dataset):
    """
    Human and Yeast PPI dataset
    """
    def __init__(self, is_train, data_type, repetition, fold, data_dir, feature_type):
        self.data_dir = data_dir
        self.feature_type = feature_type
        if is_train:
            pos_splits_file = os.path.join("data", "negativeSimTest", "nonRed",
                    "C1", "{}CV".format(data_type), "split_{}".format(repetition),
                    "{}.train.pos".format(fold))
            neg_splits_file = os.path.join("data", "negativeSimTest", "nonRed",
                    "C1", "{}CV".format(data_type), "split_{}".format(repetition),
                    "{}.train.neg".format(fold))
            pos_seq_names = self.parse_splits_file(pos_splits_file, 'pos')
            neg_seq_names = self.parse_splits_file(neg_splits_file, 'neg')
        else:
            pos_splits_file_1 = os.path.join("data", "negativeSimTest", "nonRed",
                    "C1", "{}CV".format(data_type), "split_{}".format(repetition),
                    "{}.test.pos".format(fold))
            pos_splits_file_2 = os.path.join("data", "negativeSimTest", "nonRed",
                    "C2", "{}CV".format(data_type), "split_{}".format(repetition),
                    "{}.test.pos".format(fold))
            neg_splits_file_1 = os.path.join("data", "negativeSimTest", "nonRed",
                    "C1", "{}CV".format(data_type), "split_{}".format(repetition),
                    "{}.test.neg".format(fold))
            neg_splits_file_2 = os.path.join("data", "negativeSimTest", "nonRed",
                    "C2", "{}CV".format(data_type), "split_{}".format(repetition),
                    "{}.test.neg".format(fold))
            pos_seq_names_1 = self.parse_splits_file(pos_splits_file_1, 'pos')
            neg_seq_names_1 = self.parse_splits_file(neg_splits_file_1, 'neg')
            pos_seq_names_2 = self.parse_splits_file(pos_splits_file_2, 'pos')
            neg_seq_names_2 = self.parse_splits_file(neg_splits_file_2, 'neg')
            pos_seq_names = pos_seq_names_1 + pos_seq_names_2
            neg_seq_names = neg_seq_names_1 + neg_seq_names_2
        # Ensure an even number of positive and negative samples
        if len(neg_seq_names) > len(pos_seq_names):
            neg_seq_names_idxs = np.random.choice(list(range(len(neg_seq_names))),
                    size=len(pos_seq_names), replace=False)
            neg_seq_names_ = np.array(neg_seq_names)
            neg_seq_names = neg_seq_names_[neg_seq_names_idxs.tolist()].tolist()
        print("number of positive pairs: {}".format(len(pos_seq_names)))
        print("number of negative pairs: {}".format(len(neg_seq_names)))
        self.seq_names = pos_seq_names + neg_seq_names
        np.random.shuffle(self.seq_names)
        # remove all pairs containing any seq in the ignore file
        ignore_file = os.path.join(data_dir, 'ignore.txt')
        if os.path.exists(ignore_file):
            ignore = []
            with open(ignore_file, 'r') as f:
                lines = list(map(lambda x: x.strip('\n').strip(' '), f.readlines()))
                ignore = lines
            new_seq_names = []
            for s in self.seq_names:
                if s[0] in ignore or s[1] in ignore:
                    continue
                new_seq_names.append(s)
            self.seq_names = new_seq_names

    def parse_splits_file(self, sp_file, label):
        data = []
        with open(sp_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                if label == 'neg':
                    x,y = l.split(" ")
                else:
                    x,y,_ = l.split(" ")
                data.append((x.strip('\n'), y.strip('\n'), label))
        return data

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        sample = self.seq_names[idx]
        x = np.load(os.path.join(self.data_dir, sample[0] + '.npy'))
        y = np.load(os.path.join(self.data_dir, sample[1] + '.npy'))
        label = sample[2]
        z = 1 if label == 'pos' else 0
        return {'x': np.concatenate([x,y]).astype(np.float32), 'label': z}

if __name__ == '__main__':
    d = PPIDataset('data/negativeSimTest/nonRed/C3/yeastCV/split_0/3.test.pos', 'data/negativeSimTest/nonRed/C3/yeastCV/split_0/3.test.neg', 'data/yeast_ac', 'AC')
    print(len(d))
    print(d[0])
