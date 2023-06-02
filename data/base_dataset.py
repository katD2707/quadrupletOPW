from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y, num_per_class=None):
        self.dataset = (X, y)
        self.num_per_class = num_per_class

    def create_triplet(self):
        all_class_ids = list(set(self.dataset[1]))
        label2data =  dict(zip(all_class_ids, [[]]*len(all_class_ids)))
        new_dataset = []
        for (x, y) in self.dataset:
            label2data[y].append(x)

        if self.num_per_class is None:
            for label in all_class_ids:
                for other in set(all_class_ids)-set(label):
                    x = [label2data[label][0], ]
        else:
            P = label2data[0][:self.num_per_class]
            Q = label2data[0][self.num_per_class:]
            R = []
            for other in set(all_class_ids)-set(0):
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
        return self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2]    # list type
