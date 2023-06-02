from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
import torch
from data.base_dataset import CustomDataset


class SADDataset:
    def __init__(self,
                 train_path,
                 test_path=None,
                 test_size: Optional[int, float] = 0.3,
                 num_classes=10,
                 num_per_class=None):
        self.num_classes = num_classes

        self.X_train, num_train = self.get_utterances(train_path)
        self.y_train = self.get_labels(num_train)
        if test_path is not None:
            self.X_test, num_test = self.get_utterances(test_path)
            self.y_test = self.get_labels(num_test)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=test_size, random_state=42
                                                                                    )
        self.train_data = CustomDataset(self.X_train, self.y_train, num_per_class).create_triplet()
        self.test_data = CustomDataset(self.X_test, self.y_test)

    def get_utterances(self, path):
        X = []
        utterances = []
        count = 0
        with open(path, 'r') as f:
            length_X = len([line for line in f.read().splitlines()])
        with open(path, 'r') as f:
            for line in f.read().splitlines():
                count += 1
                frame = [-abs(float(i)) if i.startswith('-') else abs(float(i)) for i in line.split()]
                if len(frame) > 0:
                    utterances.append(frame)
                    if count == length_X:
                        X.append(utterances)
                else:
                    if len(utterances) > 0:
                        X.append(utterances)
                        utterances = []

        return X, len(X)

    def get_labels(self, num_record):
        n_class = torch.Tensor(torch.arange(0, self.num_classes)).view(-1, 1)

        y = n_class.expand_as(torch.empty((self.num_classes, num_record / self.num_classes))).contiguous().view(-1).tolist()
        return y

    def get_train_loader(self):
        return DataLoader(self.train_data,
                          )


class Action3DDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class Activity3DDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
