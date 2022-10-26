import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Data:
    def __init__(self, plot_path):
        self.train_set = None
        self.validate_set = None
        self.plot_path = plot_path

    def train_loader(self, alpha, n_clients, batch_size, flag=True):
        labels = np.array(self.train_set.targets)
        split_idx = self.split_non_iid(labels, alpha=alpha, n_clients=n_clients)
        data_loader = []
        if flag:
            plt.figure(figsize=(20, 3))
            plt.hist([labels[idc] for idc in split_idx], stacked=True,
                     bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
                     label=["Client {}".format(i + 1) for i in range(n_clients)], rwidth=0.5)
            plt.xticks(np.arange(len(self.train_set.classes)), self.train_set.classes)
            plt.legend()
            plt.savefig(self.plot_path + "/" + self.__class__.__name__ + "_data_distribution.png")
        sz = [len(self.train_set)]
        targets = torch.zeros(sz)
        sz += [i for i in self.train_set[0][0].size()]
        feature = torch.zeros(sz)
        for i, v in enumerate(self.train_set):
            feature[i] = v[0]
            targets[i] = v[1]
        client_nums = []
        total = 0
        for i in split_idx:
            total += len(i)
            client_nums.append(len(i))
            data_loader.append(
                DataLoader(dataset=DatasetNonIID(feature[i], targets[i]),
                           batch_size=batch_size,
                           shuffle=True), )
        return data_loader, client_nums, total

    def validate_loader(self, batch_size):
        return DataLoader(self.validate_set, batch_size=batch_size, shuffle=False)

    @staticmethod
    def split_non_iid(train_labels, alpha, n_clients):
        n_classes = train_labels.max() + 1
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
        class_idcs = [np.argwhere(train_labels == y).flatten()
                      for y in range(n_classes)]
        client_idcs = [[] for _ in range(n_clients)]
        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]
        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
        return client_idcs


class DatasetNonIID(Dataset):
    def __init__(self, data, targets):
        super(DatasetNonIID, self).__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], int(self.targets[index].item())
