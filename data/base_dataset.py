from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, X, y, num_per_class=None):
        self.dataset = (X, y)
        self.num_per_class = num_per_class

    def create_triplet(self):
        all_class_ids = list(set(self.dataset[1]))
        label2data =  dict(zip(all_class_ids, [[]]*len(all_class_ids)))
        new_dataset = []
        for i in range(len(self.dataset[0])):
            label2data[self.dataset[1][i]].append(self.dataset[0][i])

        if self.num_per_class is None:
            for label in all_class_ids:
                for other in set(all_class_ids)- {label}:
                    x = [label2data[label][0], ]
        else:
            P = label2data[0][:self.num_per_class]
            Q = label2data[0][self.num_per_class:]
            R = []
            for other in set(all_class_ids)- {0}:
                R += label2data[other]
            for i in P:
                for j in Q:
                    for k in R:
                        record = [i, j, k]
                        new_dataset.append(record)
        self.dataset = new_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.Tensor(self.dataset[idx][0]), torch.Tensor(self.dataset[idx][1]), torch.Tensor(self.dataset[idx][2])    # list type
